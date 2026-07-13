# -*- coding: utf-8 -*-
"""
vector_store/retriever.py

search_store(query, store_dir, top_k, embedding_fn) -> List[dict]

加载 index.faiss + meta.jsonl，对 query 做 embedding 后检索 top_k 条结果。
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


def load_store(store_dir: str) -> Tuple[object, List[dict]]:
    """
    加载一个向量库。

    Args:
        store_dir: 向量库目录（包含 index.faiss, meta.jsonl, config.json）。

    Returns:
        (faiss_index, records_list)

    Raises:
        FileNotFoundError: store_dir 不存在或缺少必要文件。
    """
    if faiss is None:
        raise ImportError(
            "faiss is not installed. Install faiss-cpu in the active conda "
            "environment to use multi-vector retrieval."
        )

    store_dir = os.path.normpath(store_dir)

    if not os.path.isdir(store_dir):
        raise FileNotFoundError(f"向量库目录不存在：{store_dir}")

    index_path = os.path.join(store_dir, "index.faiss")
    meta_path = os.path.join(store_dir, "meta.jsonl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"缺少 index.faiss：{index_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"缺少 meta.jsonl：{meta_path}")

    index = faiss.read_index(index_path)
    print(f"[INFO] Loaded index: {index.ntotal} vectors, dim={index.d}")

    records: List[dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"[INFO] Loaded meta: {len(records)} records")
    return index, records


def search_store(
    query: str,
    store_dir: str,
    top_k: int,
    embedding_fn: Callable[[str], np.ndarray],
) -> List[dict]:
    """
    在单个向量库中检索 top_k 条最相似记录。

    Args:
        query: 查询文本。
        store_dir: 向量库目录。
        top_k: 返回结果数。
        embedding_fn: 查询 embedding 函数，fn(str) -> np.ndarray (1D)。

    Returns:
        List[dict]，每条包含原 record 字段 + "score"。
    """
    index, records = load_store(store_dir)

    if not records:
        print(f"[WARN] 向量库为空：{store_dir}")
        return []

    # 嵌入 query
    query_vec = np.asarray(embedding_fn(query), dtype="float32")
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    if query_vec.shape[1] != index.d:
        # 可能缓存用的不同模型，尝试重新嵌入
        raise ValueError(
            f"Query embedding 维度 ({query_vec.shape[1]}) 与索引维度 ({index.d}) 不一致。\n"
            f"向量库：{store_dir}\n"
            "可能原因：索引用了一个模型，查询用了另一个模型。\n"
            "建议：用 --force 重建向量库，或确保使用同一 embedding 后端。"
        )

    faiss.normalize_L2(query_vec)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec, k)

    results: List[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(records):
            continue
        record = dict(records[idx])
        record["score"] = float(score)
        results.append(record)

    return results
