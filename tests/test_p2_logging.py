"""CPU extraction must not initialize or reset another task's CUDA context."""
import json

from tools import logger


def test_cpu_log_never_queries_cuda(monkeypatch, tmp_path):
    def forbidden_cuda():
        raise AssertionError("CPU logging must not touch CUDA")

    monkeypatch.setattr(logger, "_cuda", forbidden_cuda)
    with logger.RunLogger("cpu_extract", log_dir=tmp_path, track_cuda=False) as log:
        log.metric("n_images", 2)
    record = json.loads((tmp_path / "runs.jsonl").read_text(encoding="utf-8"))
    assert record["status"] == "ok"
    assert record["n_images"] == 2
    assert record["gpu_mem_peak_MB"] == 0
