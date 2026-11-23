"""P5 schema, drift references, partial delivery and full synthetic integration."""
import json
import argparse
import shutil

import numpy as np
import pandas as pd
import pytest

from eval.drift import cosine_summary, evaluate_drift
from eval.eval_retrieval import build_drop_table
from eval.p5_common import (DRIFT_COLUMNS, METRICS, RETRIEVAL_COLUMNS, FeatureKey,
                            add_input_arguments, discover_features, inspect_features, load_features,
                            resolve_inputs, REPO_ROOT)
from eval.run_p5 import run_pipeline
from tools.make_dummy_features import make


@pytest.fixture(scope="module")
def full_dummy(tmp_path_factory):
    root = tmp_path_factory.mktemp("p5_features")
    assert make(str(root), n=12) == 128
    return root


def pretrained_subset(full_dummy, tmp_path):
    root = tmp_path / "input"
    source = full_dummy / "clip" / "pretrained"
    destination = root / "clip" / "pretrained"
    destination.mkdir(parents=True)
    for name in ("flickr_val_clean.npz", "flickr_val_noise.npz"):
        shutil.copy2(source / name, destination / name)
    return root


def test_known_cosines_and_population_std():
    current = np.array([[1, 0], [0, 1], [-1, 0]], dtype=float)
    reference = np.tile([1, 0], (3, 1))
    result = cosine_summary(current, reference)
    assert result == pytest.approx(dict(mean=0, std=np.sqrt(2 / 3), p5=-0.9, p50=0, p95=0.9, n=3))
    identity = cosine_summary(current, current)
    assert identity["mean"] == 1 and identity["std"] == 0


def test_full_synthetic_pipeline(full_dummy, tmp_path):
    out = tmp_path / "results"
    state = run_pipeline(full_dummy, out, no_plots=True, block_size=7)
    assert state["status"] == "completed"
    assert state["coverage"] == dict(available_files=128, expected_files=128, complete=True)
    assert (state["retrieval_val_rows"], state["retrieval_test_rows"], state["drop_rows"], state["drift_rows"]) == (80, 48, 576, 448)
    val, test = pd.read_csv(out / "retrieval_val.csv"), pd.read_csv(out / "retrieval_test.csv")
    assert list(val.columns) == list(test.columns) == RETRIEVAL_COLUMNS
    assert set(val["setting"]) == {"pretrained", "lit", "lora_r8", "lora_r16"}
    assert {"noise_lo", "noise_hi", "occl_lo", "occl_hi"}.issubset(set(val["cond"]))
    assert set(test["cond"]) == {"clean", "noise", "occl"}
    drift = pd.read_csv(out / "drift_cosine.csv")
    assert list(drift.columns) == DRIFT_COLUMNS
    assert not drift.duplicated(["dataset", "split", "model", "setting", "cond", "modality", "ref"]).any()
    assert set(drift["split"]) == {"val", "test"}
    identity = drift[(drift["setting"] == "pretrained") & (drift["ref"] == "same_cond")]
    assert (identity["mean"] == 1).all() and (identity["std"] == 0).all()
    lit = drift[(drift["setting"] == "lit") & (drift["modality"] == "image") & (drift["ref"] == "same_cond")]
    assert (lit["mean"] == 1).all()
    lit_tuned = drift[(drift["setting"] == "lit") & (drift["modality"] == "text")]
    assert (lit_tuned["mean"] < 1).all()
    assert set(drift["ref"]) == {"same_cond", "pretrained_clean", "self_clean"}
    assert json.loads((out / "p5_drift_missing_references.json").read_text(encoding="utf-8")) == []
    assert "实验数字记录" in (out / "p5_observations.md").read_text(encoding="utf-8")


def test_partial_features_do_not_invent_finetuned_results(full_dummy, tmp_path):
    root = pretrained_subset(full_dummy, tmp_path)
    out = tmp_path / "results"
    state = run_pipeline(root, out, no_plots=True)
    assert state["coverage"]["available_files"] == 2 and not state["coverage"]["complete"]
    val = pd.read_csv(out / "retrieval_val.csv")
    assert len(val) == 2 and set(val["setting"]) == {"pretrained"}
    assert pd.read_csv(out / "retrieval_test.csv").empty
    assert len(pd.read_csv(out / "retrieval_drop.csv")) == 6


def test_complete_mode_refuses_partial_input(full_dummy, tmp_path):
    root = pretrained_subset(full_dummy, tmp_path)
    out = tmp_path / "results"
    with pytest.raises(ValueError, match="missing"):
        run_pipeline(root, out, require_complete=True, no_plots=True)
    assert not (out / "retrieval_val.csv").exists()
    assert json.loads((out / "p5_run.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_misaligned_ids_fail_before_evaluation(full_dummy, tmp_path):
    root = pretrained_subset(full_dummy, tmp_path)
    noise = root / "clip/pretrained/flickr_val_noise.npz"
    with np.load(noise) as data:
        arrays = {key: data[key] for key in data.files}
    arrays["img_ids"] = arrays["img_ids"][::-1]
    np.savez(noise, **arrays)
    with pytest.raises(ValueError, match="IDs/order"):
        inspect_features(root, discover_features(root))


def test_official_split_check(full_dummy, tmp_path):
    key = FeatureKey("clip", "pretrained", "flickr", "val", "clean")
    data = load_features(full_dummy, key)
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    split_path = split_dir / "flickr_split.json"
    split_path.write_text(json.dumps({"val": data.ids.tolist()}), encoding="utf-8")
    load_features(full_dummy, key, split_dir)
    split_path.write_text(json.dumps({"val": data.ids[::-1].tolist()}), encoding="utf-8")
    # A different path avoids the immutable official-split cache.
    other_dir = tmp_path / "bad_splits"
    other_dir.mkdir()
    shutil.copy2(split_path, other_dir / split_path.name)
    with pytest.raises(ValueError, match="official split"):
        load_features(full_dummy, key, other_dir)


def test_zero_baseline_drop_has_no_division_by_zero():
    rows = []
    for cond, score in (("clean", 0.0), ("noise", 20.0)):
        rows.append(dict(dataset="flickr", split="val", model="clip", setting="pretrained", cond=cond,
                         **{metric: score for metric in METRICS}))
    result = build_drop_table(pd.DataFrame(rows))
    assert (result["drop_pp"] == -20).all()
    assert result["drop_relative_pct"].isna().all()


def test_missing_pretrained_reference_is_reported(full_dummy, tmp_path):
    key = FeatureKey("clip", "lit", "flickr", "val", "clean")
    frame, missing = evaluate_drift(full_dummy, tmp_path, keys=[key])
    assert frame.empty and len(missing) == 1
    assert "pretrained" in missing[0]["missing_reference"]


def test_missing_clean_file_fails_with_clear_error(full_dummy, tmp_path):
    root = pretrained_subset(full_dummy, tmp_path)
    (root / "clip/pretrained/flickr_val_clean.npz").unlink()
    with pytest.raises(ValueError, match="Missing clean"):
        inspect_features(root, discover_features(root))


def test_complete_clip_stage_without_siglip_dependency(full_dummy, tmp_path):
    out = tmp_path / "clip_results"
    state = run_pipeline(full_dummy, out, model="clip", require_complete=True, no_plots=True)
    assert state["model_scope"] == "clip"
    assert state["coverage"] == dict(available_files=64, expected_files=64, complete=True)
    assert (state["retrieval_val_rows"], state["retrieval_test_rows"], state["drop_rows"], state["drift_rows"]) == (40, 24, 288, 224)
    for name in ("retrieval_val.csv", "retrieval_test.csv", "retrieval_drop.csv", "drift_cosine.csv"):
        assert set(pd.read_csv(out / name)["model"]) == {"clip"}
    report = json.loads((out / "p5_feature_check.json").read_text(encoding="utf-8"))
    assert report["expected_models"] == ["clip"] and report["missing"] == []
    assert "单模型阶段" in (out / "p5_observations.md").read_text(encoding="utf-8")


def test_clip_discovery_ignores_unselected_model_files(full_dummy, tmp_path):
    root = pretrained_subset(full_dummy, tmp_path)
    unselected = root / "siglip" / "bad_file.npz"
    unselected.parent.mkdir()
    unselected.write_bytes(b"not an NPZ")
    assert len(discover_features(root, model="clip")) == 2
    with pytest.raises(ValueError, match="Unexpected NPZ"):
        discover_features(root)


def test_single_model_default_output_is_separate():
    parser = add_input_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--model", "clip"])
    _, out, _ = resolve_inputs(args)
    assert out == REPO_ROOT / "results" / "p5_clip"
    args = parser.parse_args(["--model", "clip", "--no-split-check"])
    with pytest.raises(ValueError, match="separate --out-dir"):
        resolve_inputs(args)
