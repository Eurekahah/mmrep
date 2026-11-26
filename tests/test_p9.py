"""P9 regression checks: paired identities, joint fitting, source values and run provenance."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.make_dummy_features import make
from tools.logger import RunLogger
from tools.summarize_logs import summarize
from viz.make_report_tables import build_tables, format_value
from viz.tsne_umap import Group, groups, prepare_group, reduce_joint, sample_indices


@pytest.fixture(scope="module")
def features(tmp_path_factory):
    root = tmp_path_factory.mktemp("p9_features")
    make(root, n=40)
    return root


def test_sampling_uses_ids():
    ids = np.array(["b", "a", "d", "c"])
    rev = ids[::-1]
    assert np.array_equal(ids[sample_indices(ids, 3, 42)], rev[sample_indices(rev, 3, 42)])
    with pytest.raises(ValueError): sample_indices(ids, 5, 42)


def test_complete_group_coverage():
    planned = groups(("clip", "siglip"), ("flickr", "coco"))
    assert len(planned) == 29  # 28 main comparisons + one joint seven-condition strength group
    assert sum(g.kind == "strength" for g in planned) == 1
    assert len(groups(("clip",), ("flickr",), "test", "strength")) == 0


def test_settings_joint_and_first_caption(features):
    g = groups(("clip",), ("coco",), scope="settings")[0]
    x, frame, _ = prepare_group(features, g, 40)
    assert x.shape == (320, 512)
    assert np.allclose(np.linalg.norm(x, axis=1), 1)
    assert (frame[frame.modality == "text"].source_row % 5 == 0).all()
    for setting in ("pretrained", "lit", "lora_r8", "lora_r16"):
        assert frame[(frame.setting == setting) & (frame.modality == "image")].paired_line.sum() == 30
    assert frame.groupby("setting").img_id.apply(set).apply(lambda s: s == set(frame.img_id)).all()


def test_conditions_share_text(features):
    g = groups(("clip",), ("flickr",), scope="conditions")[0]
    x, frame, _ = prepare_group(features, g, 40)
    assert x.shape == (160, 512)
    assert (frame.modality == "text").sum() == 40
    assert set(frame[frame.modality == "text"].cond) == {"clean"}


def test_reject_misaligned_ids(features, tmp_path):
    import shutil
    shutil.copytree(features / "clip" / "pretrained", tmp_path / "clip" / "pretrained")
    path = tmp_path / "clip/pretrained/flickr_val_noise.npz"
    with np.load(path) as archive: data = dict(archive)
    data["img_ids"] = data["img_ids"][::-1]
    np.savez(path, **data)
    g = groups(("clip",), ("flickr",), scope="conditions")[0]
    with pytest.raises(ValueError, match="IDs/order"):
        prepare_group(tmp_path, g, 40)


@pytest.mark.parametrize("method", ["tsne", "umap"])
def test_real_reducer_is_finite_and_repeatable(method):
    from threadpoolctl import threadpool_limits
    x = np.random.default_rng(42).normal(size=(64, 64)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    with threadpool_limits(limits=1):
        first, energy = reduce_joint(x, method, max_iter=300)
        second, _ = reduce_joint(x, method, max_iter=300)
    assert first.shape == (64, 2)
    assert np.array_equal(first, second)
    assert 0 < energy <= 1.00001


def test_table_rounding_and_provenance(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    pd.DataFrame([dict(model="clip", setting="pretrained", top1_mean=.7636666667,
                       top1_std=.0004027682)]).to_csv(source / "cifar100_summary.csv", index=False)
    result = build_tables(source, tmp_path / "tables")
    display = pd.read_csv(tmp_path / "tables/cifar100.csv")
    assert display.top1_mean[0] == "76.4%"
    assert result["cells"][0]["raw_value"] == .7636666667
    assert not result["complete"]
    assert "svd" in result["missing"]
    assert format_value("gap", .123456) == "0.1235"
    assert format_value("top1_energy", .512345) == "0.5123"
    assert format_value("i2t_R1", 76.4) == "76.4%"
    assert format_value("i2t_R1", .8) == "0.8%"  # Do not guess units from magnitude.
    with pytest.raises(ValueError, match="must be in"):
        format_value("top1", 76.4)


def test_summary_conflict_is_not_silently_accepted(tmp_path):
    row = dict(dataset="flickr", model="clip", setting="pretrained", cond="clean", i2t_R1=70.)
    pd.DataFrame([row]).to_csv(tmp_path / "retrieval_val.csv", index=False)
    with pd.ExcelWriter(tmp_path / "summary_table.xlsx") as writer:
        pd.DataFrame([{**row, "i2t_R1": 80.}]).to_excel(writer, sheet_name="main", index=False)
    with pytest.raises(ValueError, match="conflicts"):
        build_tables(tmp_path, tmp_path / "tables")


def test_workbook_fallback_and_strength(tmp_path):
    row = dict(dataset="flickr", model="clip", setting="pretrained", cond="noise_lo", i2t_R1=70., gap=.123456)
    with pd.ExcelWriter(tmp_path / "summary_table.xlsx") as writer:
        pd.DataFrame([row]).to_excel(writer, sheet_name="main", index=False)
        pd.DataFrame([row]).to_excel(writer, sheet_name="strength", index=False)
    result = build_tables(tmp_path, tmp_path / "tables")
    assert {"retrieval", "modality_gap", "strength"}.issubset({t["name"] for t in result["tables"]})
    assert pd.read_csv(tmp_path / "tables/strength.csv").i2t_R1[0] == "70.0%"


def test_runlogger_metadata_and_duplicates(tmp_path):
    logs, ckpt = tmp_path / "logs", tmp_path / "ckpt"
    with RunLogger("train_lora", model="clip", setting="lora_r8", seed=42, log_dir=logs, track_cuda=False):
        pass
    record = (logs / "runs.jsonl").read_text()
    (logs / "P4_runs.jsonl").write_text(record + record)
    meta_dir = ckpt / "clip/lora_r8"
    meta_dir.mkdir(parents=True)
    (meta_dir / "meta.json").write_text(json.dumps(dict(epochs=5, batch_size=128, seed=42, wall_sec=999)))
    info = summarize(logs, ckpt, tmp_path / "out")
    frame = pd.read_csv(tmp_path / "out/compute_log.csv")
    assert len(frame) == 1
    assert frame.epoch[0] == 5 and frame.batch[0] == 128
    assert frame.wall_sec[0] != 999
    assert info["exact_duplicates_removed"] == 1
    assert set(info["missing_log_owners"]) == {"P2", "P3", "P6"}


def test_bad_log_reports_source_line(tmp_path):
    (tmp_path / "P2_runs.jsonl").write_text('{"name":"ok"}\nnot json\n')
    with pytest.raises(ValueError, match="P2_runs.jsonl:2"):
        summarize(tmp_path, tmp_path / "ckpt", tmp_path / "out")
