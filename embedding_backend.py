# -*- coding: utf-8 -*-
"""
embedding_backend.py

统一处理 embedding API 失败后的本地回退：
1. API 报错时自动加载 / 下载 sentence-transformers 本地 embedding 模型。
2. 默认使用 BAAI/bge-small-en-v1.5，小模型 CPU 可跑，内存远低于 16GB。
3. 所有本地模型统一放到 ./models/embedding，避免每次重复下载。
4. main_MedRAG.py 和 KG_Retrieve.py 共用同一套本地模型与状态，避免维度不一致。
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


DEFAULT_LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LOCAL_FALLBACK_MODELS = [
    "BAAI/bge-small-en-v1.5",           # 约 100MB 级别，384 维，16GB 内存机器很稳
    "sentence-transformers/all-MiniLM-L6-v2",  # 更小兜底模型，384 维
]
DEFAULT_CACHE_DIR = "./models/embedding"
DEFAULT_DEVICE = "cpu"
DEFAULT_BATCH_SIZE = 16
DEFAULT_MAX_MEMORY_GB = 16.0

# 粗略估计值，用于防止误配超大模型。不是精确运行时占用，但足够做安全阈值过滤。
MODEL_MEMORY_PROFILE_GB = {
    "BAAI/bge-small-en-v1.5": 1.0,
    "sentence-transformers/all-MiniLM-L6-v2": 0.8,
    "BAAI/bge-base-en-v1.5": 2.0,
    "BAAI/bge-large-en-v1.5": 4.0,
    "BAAI/bge-m3": 5.0,
}

_EMBEDDING_STATE: Dict[str, Any] = {
    "backend": "remote",
    "model": None,
    "dim": None,
}
_LOCAL_MODEL_CACHE: Dict[str, Any] = {}
_LAST_LOCAL_REASON: Optional[str] = None


def _get_auth_value(name: str, default: Any) -> Any:
    """从 authentication.py 读取可选配置；没有就使用默认值。"""
    try:
        auth = importlib.import_module("authentication")
        return getattr(auth, name, default)
    except Exception:
        return default


def safe_model_name(model_name: str) -> str:
    return (
        str(model_name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(".", "_")
        .replace(":", "_")
    )


def get_embedding_state() -> Dict[str, Any]:
    return dict(_EMBEDDING_STATE)


def set_embedding_state(backend: Optional[str] = None, model: Optional[str] = None, dim: Optional[int] = None) -> None:
    if backend:
        _EMBEDDING_STATE["backend"] = backend
    if model:
        _EMBEDDING_STATE["model"] = model
    if dim is not None:
        _EMBEDDING_STATE["dim"] = int(dim)


def set_embedding_state_from_meta(meta: Optional[Dict[str, Any]]) -> None:
    if not isinstance(meta, dict):
        return
    set_embedding_state(
        backend=meta.get("actual_backend"),
        model=meta.get("actual_model"),
        dim=meta.get("dim"),
    )


def _unique(items: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def get_local_embedding_models() -> List[str]:
    preferred = _get_auth_value("local_embedding_model", DEFAULT_LOCAL_EMBEDDING_MODEL)
    configured = _get_auth_value("local_embedding_fallback_models", None)

    if configured is None:
        candidates = [preferred] + DEFAULT_LOCAL_FALLBACK_MODELS
    else:
        candidates = [preferred] + list(configured) + DEFAULT_LOCAL_FALLBACK_MODELS

    return _unique(candidates)


def get_local_cache_dir() -> str:
    cache_dir = _get_auth_value("local_embedding_cache_dir", DEFAULT_CACHE_DIR)
    cache_dir = os.path.abspath(str(cache_dir))
    os.makedirs(cache_dir, exist_ok=True)

    # transformers / huggingface_hub 会使用这些目录缓存模型。
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

    hf_endpoint = str(_get_auth_value("hf_endpoint", "") or "").strip()
    if hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

    return cache_dir


def _get_cached_model_path(model_name: str, cache_dir: str) -> Optional[str]:
    cache_name = "models--" + str(model_name).replace("/", "--")
    snapshots_dir = os.path.join(cache_dir, cache_name, "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    candidates = []
    for entry in os.listdir(snapshots_dir):
        snapshot_path = os.path.join(snapshots_dir, entry)
        if not os.path.isdir(snapshot_path):
            continue
        if os.path.exists(os.path.join(snapshot_path, "config.json")):
            candidates.append(snapshot_path)

    if not candidates:
        return None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates[0]


def _get_max_memory_gb() -> float:
    try:
        return float(_get_auth_value("local_embedding_max_memory_gb", DEFAULT_MAX_MEMORY_GB))
    except Exception:
        return DEFAULT_MAX_MEMORY_GB


def _get_batch_size() -> int:
    try:
        batch_size = int(_get_auth_value("local_embedding_batch_size", DEFAULT_BATCH_SIZE))
    except Exception:
        batch_size = DEFAULT_BATCH_SIZE
    return max(1, min(batch_size, 32))


def _get_device() -> str:
    # 为了保证 16GB 内存机器稳定，默认强制 CPU。需要 GPU 时可在 authentication.py 改 local_embedding_device。
    return str(_get_auth_value("local_embedding_device", DEFAULT_DEVICE) or DEFAULT_DEVICE)


def _estimated_model_memory_gb(model_name: str) -> float:
    return float(MODEL_MEMORY_PROFILE_GB.get(model_name, 6.0))


def _is_model_within_memory_budget(model_name: str) -> bool:
    max_memory_gb = _get_max_memory_gb()
    estimated = _estimated_model_memory_gb(model_name)
    if estimated > max_memory_gb:
        print(
            f"[WARN] Skip local embedding model {model_name}: "
            f"estimated memory {estimated:.1f}GB > limit {max_memory_gb:.1f}GB"
        )
        return False
    return True


def _ensure_sentence_transformers_available():
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        return
    except ImportError:
        auto_install = bool(_get_auth_value("auto_install_sentence_transformers", True))
        if not auto_install:
            raise RuntimeError(
                "本地 embedding 回退需要 sentence-transformers，但当前环境未安装。\n"
                "请执行：python -m pip install sentence-transformers"
            )

        print("[WARN] 未检测到 sentence-transformers，正在自动安装...")
        try:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "sentence-transformers>=3.1.1,<4.0.0",
            ])
        except Exception as e:
            raise RuntimeError(
                "自动安装 sentence-transformers 失败。请手动执行：\n"
                "  python -m pip install sentence-transformers\n"
                "或先执行：\n"
                "  python -m pip install -r requirements.txt"
            ) from e


def _load_local_sentence_transformer(model_name: str):
    if model_name in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_name]

    if not _is_model_within_memory_budget(model_name):
        raise RuntimeError(f"模型 {model_name} 超过本地 embedding 内存限制。")

    _ensure_sentence_transformers_available()
    from sentence_transformers import SentenceTransformer

    cache_dir = get_local_cache_dir()
    device = _get_device()
    cached_model_path = _get_cached_model_path(model_name, cache_dir)
    model_name_or_path = cached_model_path or model_name
    local_files_only = cached_model_path is not None

    print(f"[INFO] Loading local embedding model: {model_name}")
    print(f"[INFO] Local model cache dir: {cache_dir}")
    print(f"[INFO] Local embedding device: {device}, batch_size={_get_batch_size()}")
    if cached_model_path:
        print(f"[INFO] Using cached local model snapshot: {cached_model_path}")
    print("[INFO] 如果本地缓存不存在，sentence-transformers 会自动下载模型。")

    model = SentenceTransformer(
        model_name_or_path,
        cache_folder=cache_dir,
        device=device,
        local_files_only=local_files_only,
    )
    _LOCAL_MODEL_CACHE[model_name] = model
    return model


def local_sentence_transformer_embeddings(texts: List[str], reason: str = "remote embedding failed") -> np.ndarray:
    """
    生成本地 embedding。
    - 第一次运行会自动下载模型到 ./models/embedding。
    - 默认模型 bge-small-en-v1.5，CPU 运行，内存远低于 16GB。
    - 所有输出统一 float32 + L2 normalize，适合 FAISS cosine/IP 检索。
    """
    if isinstance(texts, str):
        texts = [texts]

    clean_texts = [str(t or "empty medical text") for t in texts]
    last_error: Optional[BaseException] = None

    global _LAST_LOCAL_REASON
    if reason != _LAST_LOCAL_REASON:
        print(f"[WARN] Switch to local embedding because: {reason}")
        _LAST_LOCAL_REASON = reason

    for model_name in get_local_embedding_models():
        if not _is_model_within_memory_budget(model_name):
            continue

        try:
            model = _load_local_sentence_transformer(model_name)
            vectors = model.encode(
                clean_texts,
                batch_size=_get_batch_size(),
                show_progress_bar=len(clean_texts) > 1,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            vectors = np.asarray(vectors, dtype="float32")
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            set_embedding_state(
                backend="local",
                model=model_name,
                dim=int(vectors.shape[1]),
            )
            return vectors
        except Exception as e:
            last_error = e
            print(f"[WARN] Local embedding model failed: {model_name}. Error: {e}")

    raise RuntimeError(
        "所有本地 embedding 模型都加载失败。\n"
        "建议先执行：python -m pip install -r requirements.txt\n"
        "如果 Hugging Face 下载失败，可以先设置镜像或手动下载模型到 ./models/embedding。\n"
        "默认推荐模型：BAAI/bge-small-en-v1.5，内存占用远低于 16GB。"
    ) from last_error


def embedding_dim(vectors: np.ndarray) -> Optional[int]:
    arr = np.asarray(vectors)
    if arr.ndim == 2 and arr.shape[1] > 0:
        return int(arr.shape[1])
    return None
