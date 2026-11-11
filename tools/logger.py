"""统一日志：记录耗时、峰值显存、种子，写到 logs/ 下。

用法：
    from tools.logger import RunLogger
    with RunLogger("extract", model="clip", setting="pretrained", seed=42) as log:
        log.info("start")
        ...
        log.metric("n_images", 1000)
退出时自动写入 wall_sec 和 gpu_mem_peak_MB，日志文件是 logs/{name}_{model}_{setting}.log，
同时追加一行 JSON 到 logs/runs.jsonl，方便汇总。
CPU 任务传 track_cuda=False，完全跳过 CUDA 查询和峰值重置；默认行为保持兼容。
"""
import json
import os
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cuda():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


class RunLogger:
    def __init__(self, name, model="", setting="", seed=None, log_dir=None,
                 track_cuda=True, **extra):
        self.name, self.model, self.setting, self.seed = name, model, setting, seed
        self.extra = dict(extra)
        self.metrics = {}
        self.track_cuda = track_cuda
        self.log_dir = log_dir or os.path.join(REPO_ROOT, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        tag = "_".join(x for x in (name, model, setting) if x)
        self.path = os.path.join(self.log_dir, f"{tag}.log")

    def info(self, msg):
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def metric(self, key, value):
        self.metrics[key] = value
        self.info(f"{key} = {value}")

    def __enter__(self):
        t = _cuda() if self.track_cuda else None
        if t is not None:
            t.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        self.info(f"START {self.name} model={self.model} setting={self.setting} seed={self.seed}")
        return self

    def __exit__(self, exc_type, exc, tb):
        wall = time.time() - self.t0
        t = _cuda() if self.track_cuda else None
        peak = round(t.cuda.max_memory_allocated() / 2**20, 1) if t is not None else 0.0
        rec = {"time": datetime.now().isoformat(timespec="seconds"), "name": self.name,
               "model": self.model, "setting": self.setting, "seed": self.seed,
               "wall_sec": round(wall, 2), "gpu_mem_peak_MB": peak,
               "status": "error" if exc_type else "ok", **self.extra, **self.metrics}
        self.info(f"END wall_sec={rec['wall_sec']} gpu_mem_peak_MB={peak} status={rec['status']}")
        with open(os.path.join(self.log_dir, "runs.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return False
