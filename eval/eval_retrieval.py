"""Batch retrieval and corruption drops: python -m eval.eval_retrieval."""
from __future__ import annotations

import argparse
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .p5_common import (DROP_COLUMNS, METRICS, RETRIEVAL_COLUMNS, add_input_arguments,
                        assert_aligned, cli_main, discover_features, load_features,
                        resolve_inputs, sort_key, write_csv)
from .retrieval import recall_at_k


def build_drop_table(retrieval):
    """Absolute drops are percentage points; relative drops are percentages."""
    rows = []
    groups = ["dataset", "model", "setting", "split"]
    for _, group in retrieval.groupby(groups, sort=False):
        clean = group[group["cond"] == "clean"]
        if len(clean) != 1:
            raise ValueError("Each retrieval group must have exactly one clean row")
        clean = clean.iloc[0]
        for row in group[group["cond"] != "clean"].to_dict("records"):
            for metric in METRICS:
                reference, perturbed = float(clean[metric]), float(row[metric])
                rows.append({**{k: row[k] for k in ("dataset", "model", "setting", "cond", "split")},
                             "metric": metric, "clean_recall": reference,
                             "perturbed_recall": perturbed, "drop_pp": reference - perturbed,
                             "drop_relative_pct": 100 * (reference - perturbed) / reference
                             if reference != 0 else np.nan})
    return pd.DataFrame(rows, columns=DROP_COLUMNS)


def evaluate(features_root, out_dir, *, keys=None, splits_root=None, block_size=512, threads=2, model="all"):
    keys = discover_features(features_root, model) if keys is None else sorted(keys, key=sort_key)
    present, rows = set(keys), []
    with threadpool_limits(limits=threads, user_api="blas"):
        for _, group in groupby(keys, key=lambda k: k.group):
            group = list(group)
            clean_key = group[0].with_(cond="clean")
            if clean_key not in present:
                raise ValueError(f"Missing clean text file: {clean_key.relative_path}")
            clean = load_features(features_root, clean_key, splits_root)
            for key in group:
                data = clean if key.cond == "clean" else load_features(features_root, key, splits_root)
                assert_aligned(data, clean, str(key.relative_path))
                result = recall_at_k(data.img, clean.txt, clean.mapping, block_size=block_size)
                rows.append({**key.columns(), **result})
                print(f"retrieval {key.relative_path.as_posix()}: "
                      f"i2t R@1={result['i2t_R1']:.2f}%, t2i R@1={result['t2i_R1']:.2f}%", flush=True)
    frame = pd.DataFrame(rows, columns=[*RETRIEVAL_COLUMNS, "split"])
    drops = build_drop_table(frame)
    for split in ("val", "test"):
        write_csv(Path(out_dir) / f"retrieval_{split}.csv",
                  frame.loc[frame["split"] == split, RETRIEVAL_COLUMNS])
    write_csv(Path(out_dir) / "retrieval_drop.csv", drops)
    return frame, drops


def main():
    parser = add_input_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--block-size", type=int, default=512)
    args = parser.parse_args()
    root, out, splits = resolve_inputs(args)
    evaluate(root, out, splits_root=splits, block_size=args.block_size, threads=args.threads, model=args.model)
    print(f"Saved retrieval CSVs to {out}")


if __name__ == "__main__":
    cli_main(main)
