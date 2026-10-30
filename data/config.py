"""读取 configs/*.yaml 的小工具。路径都相对仓库根目录。"""
import os
from functools import lru_cache
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    with open(os.path.join(REPO_ROOT, "configs", name), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def load_models_cfg():
    return _load("models.yaml")


@lru_cache(maxsize=None)
def load_perturb_cfg():
    return _load("perturb.yaml")


def get_paths():
    """返回 (data_root, splits_dir)，MMREP_DATA_ROOT 环境变量优先。"""
    cfg = _load("paths.yaml")
    data_root = os.environ.get("MMREP_DATA_ROOT", cfg["data_root"])
    splits_dir = cfg["splits_dir"]
    if not os.path.isabs(data_root):
        data_root = os.path.normpath(os.path.join(REPO_ROOT, data_root))
    if not os.path.isabs(splits_dir):
        splits_dir = os.path.normpath(os.path.join(REPO_ROOT, splits_dir))
    return data_root, splits_dir
