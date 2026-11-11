"""Contract checks for an upstream receipt and a portable local handoff."""
import json
import hashlib
from types import SimpleNamespace
import zipfile

import pytest

from models.checkpoint_receipt import read_training_receipt
from extract.extract_features import checkpoint_hashes, weight_identity
from tools.package_p2 import verify_zip, write_zip


def test_training_receipt_validates_upstream_identity_and_history(tmp_path):
    folder = tmp_path / "clip" / "lit"
    folder.mkdir(parents=True)
    info = {"model": "clip", "setting": "lit", "epochs": 2, "lr": 0.0001,
            "batch_size": 64, "seed": 42, "trainable_params": 10, "wall_sec": 3,
            "gpu_mem_peak_MB": 100, "epoch_loss": [0.9, 0.7]}
    (folder / "trainable_state.pt").write_bytes(b"receipt fixture; loader checks real tensors")
    path = folder / "meta.json"
    path.write_text(json.dumps(info), encoding="utf-8")
    receipt = read_training_receipt(tmp_path, "clip", "lit")
    assert receipt["epochs"] == 2
    wrapper = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(_commit_hash="upstream")))
    base = {"requested_revision": None, "resolved_revision": "upstream"}
    assert weight_identity(wrapper, tmp_path, "clip", "pretrained", None, None) == base
    assert checkpoint_hashes(tmp_path, "clip", "pretrained") == {}
    assert checkpoint_hashes(tmp_path, "clip", "lit") == {
        "trainable_state.pt": hashlib.sha256((folder / "trainable_state.pt").read_bytes()).hexdigest()}
    assert weight_identity(wrapper, tmp_path, "clip", "lit", None, receipt) == {
        **base, "training_meta_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_sha256": {"trainable_state.pt":
                              hashlib.sha256((folder / "trainable_state.pt").read_bytes()).hexdigest()}}
    path.write_text(json.dumps({**info, "model": "siglip"}), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_training_receipt(tmp_path, "clip", "lit")
    path.write_text(json.dumps({**info, "epoch_loss": [0.5]}), encoding="utf-8")
    with pytest.raises(ValueError, match="loss"):
        read_training_receipt(tmp_path, "clip", "lit")
    path.write_text(json.dumps({**info, "num_samples": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="training samples"):
        read_training_receipt(tmp_path, "clip", "lit")


def test_manifest_rejects_changed_payload(tmp_path):
    payload = tmp_path / "numbers.csv"
    payload.write_bytes(b"value\n1\n")
    good = tmp_path / "valid.zip"
    write_zip(good, {"results/numbers.csv": payload}, {"stage": "fixture"})
    manifest = verify_zip(good)
    assert manifest["files"]["results/numbers.csv"]["bytes"] == payload.stat().st_size
    bad = tmp_path / "modified.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("MANIFEST.json", json.dumps(manifest))
        archive.writestr("results/numbers.csv", b"value\n2\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_zip(bad)


def test_zip_cannot_publish_bytes_that_changed_since_statistics(tmp_path):
    payload = tmp_path / "norm_stats.csv"
    payload.write_bytes(b"updated\n")
    with pytest.raises(ValueError, match="changed after"):
        write_zip(tmp_path / "handoff.zip", {"results/norm_stats.csv": payload}, {},
                  expected_hashes={"results/norm_stats.csv": "0" * 64})
    assert not (tmp_path / "handoff.zip").exists()


@pytest.mark.parametrize("archived_delta,accepted", [(1e-7, True), (1e-3, False)])
def test_freeze_handoff_uses_measured_repeat_difference(tmp_path, monkeypatch, archived_delta, accepted):
    from extract import recheck_lit
    from tools.package_p2 import review_frozen_images

    (tmp_path / "lit_frozen_check.csv").write_text(
        "model,dataset,split,cond,status,pretrained_device,max_abs\n"
        f"clip,flickr,val,noise,needs_review,cpu,{archived_delta}\n", encoding="utf-8")

    def replay(job, **args):
        return {"status": "measured", "vision_state_sha256": {"pretrained": "a", "lit": "a"},
                "measurements": {k: {"max_abs": v} for k, v in
                                 (("original_precision_pretrained_repeat", 2e-7),
                                  ("fp32_lit_vs_pretrained", 0),
                                  ("pretrained_replay_vs_archive", 0),
                                  ("lit_replay_vs_archive", 0))}}
    monkeypatch.setattr(recheck_lit, "recheck", replay)
    if accepted:
        assert len(review_frozen_images(tmp_path, tmp_path)) == 1
    else:
        with pytest.raises(ValueError, match="Measured evidence"):
            review_frozen_images(tmp_path, tmp_path)
