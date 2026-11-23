"""Paired image/text cosine drift: python -m eval.drift."""
from __future__ import annotations

import argparse
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from .p5_common import (DRIFT_COLUMNS, add_input_arguments, as_numpy, assert_aligned,
                        cli_main, discover_features, load_features, resolve_inputs,
                        sort_key, unit_rows, write_csv, write_json)


def cosine_summary(current, reference):
    """Paired cosine statistics; std is population std (ddof=0)."""
    left, right = unit_rows(current, "current", np.float64), unit_rows(reference, "reference", np.float64)
    if left.shape != right.shape:
        raise ValueError("Paired drift arrays must have identical shapes")
    values = np.clip(np.einsum("ij,ij->i", left, right), -1.0, 1.0)
    # Identical stored vectors are an exact identity control, not rounding noise.
    values[np.all(as_numpy(current) == as_numpy(reference), axis=1)] = 1.0
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    return dict(mean=float(values.mean()), std=float(values.std(ddof=0)),
                p5=float(p5), p50=float(p50), p95=float(p95), n=len(values))


def evaluate_drift(features_root, out_dir, *, keys=None, splits_root=None, model="all"):
    keys = discover_features(features_root, model) if keys is None else sorted(keys, key=sort_key)
    present, rows, missing = set(keys), [], []

    def add(key, modality, ref, current, reference):
        rows.append({**key.columns(), "modality": modality, "ref": ref,
                     **cosine_summary(current, reference)})

    for _, group in groupby(keys, key=lambda k: k.group):
        group = list(group)
        clean_key = group[0].with_(cond="clean")
        if clean_key not in present:
            raise ValueError(f"Missing clean feature file: {clean_key.relative_path}")
        clean = load_features(features_root, clean_key, splits_root)
        base_key = clean_key.with_(setting="pretrained")
        if base_key not in present:
            missing.append({"group": str(clean_key.relative_path), "missing_reference": str(base_key.relative_path)})
            continue
        base_clean = clean if clean_key.setting == "pretrained" else load_features(features_root, base_key, splits_root)
        assert_aligned(clean, base_clean, f"{clean_key.relative_path} vs pretrained clean")
        for key in group:
            data = clean if key.cond == "clean" else load_features(features_root, key, splits_root)
            assert_aligned(data, clean, str(key.relative_path))
            if key.setting == "pretrained":
                add(key, "image", "same_cond", data.img, data.img)
                if key.cond == "clean":
                    add(key, "text", "same_cond", clean.txt, clean.txt)
                else:
                    add(key, "image", "self_clean", data.img, base_clean.img)
            else:
                reference_key = key.with_(setting="pretrained")
                if reference_key in present:
                    base_same = base_clean if key.cond == "clean" else load_features(features_root, reference_key, splits_root)
                    assert_aligned(data, base_same, f"{key.relative_path} vs pretrained same condition")
                    add(key, "image", "same_cond", data.img, base_same.img)
                else:
                    missing.append({"file": str(key.relative_path), "missing_reference": str(reference_key.relative_path)})
                add(key, "image", "pretrained_clean", data.img, base_clean.img)
                # Image corruptions do not modify text: both text refs use pretrained clean.
                for ref in ("same_cond", "pretrained_clean"):
                    add(key, "text", ref, clean.txt, base_clean.txt)
            print(f"drift {key.relative_path.as_posix()}", flush=True)
    frame = pd.DataFrame(rows, columns=DRIFT_COLUMNS)
    write_csv(Path(out_dir) / "drift_cosine.csv", frame)
    write_json(Path(out_dir) / "p5_drift_missing_references.json", missing)
    return frame, missing


def main():
    parser = add_input_arguments(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    root, out, splits = resolve_inputs(args)
    frame, missing = evaluate_drift(root, out, splits_root=splits, model=args.model)
    print(f"Saved {len(frame)} drift rows to {out}; missing references: {len(missing)}")


if __name__ == "__main__":
    cli_main(main)
