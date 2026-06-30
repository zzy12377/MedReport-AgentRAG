# -*- coding: utf-8 -*-
"""
vector_store/builder.py

build_faiss_store(records, output_dir, embedding_fn, batch_size=8, force=False)

把标准化记录列表转为 FAISS 索引 + 元数据，存入 output_dir：
  - index.faiss   (FAISS 向量索引)
  - meta.jsonl    (每行一条标准化 record JSON)
  - config.json   (source, dim, num_records, embedding_backend, model)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Callable, List

import faiss
import numpy as np
from tqdm import tqdm

from embedding_backend import get_embedding_state


def build_faiss_store(
    records: List[dict],
    output_dir: str,
    embedding_fn: Callable[[List[str]], np.ndarray],
    batch_size: int = 8,
    force: bool = False,
) -> None:
    """
    构建 FAISS 向量库并保存到 output_dir。

    Args:
        records: 标准化 record 列表（来自 adapters）。
        output_dir: 输出目录，如 vector_db/ddxplus_cases/。
        embedding_fn: embedding 函数，签名为 fn(texts: List[str]) -> np.ndarray。
        batch_size: 每批传给 embedding_fn 的文本数。
        force: True 则覆盖已有输出目录。

    Raises:
        ValueError: records 为空或 embedding 维度异常。
    """
    output_dir = os.path.normpath(output_dir)

    if not records:
        raise ValueError("records 为空，无法构建向量库。请先确认数据源有可用记录。")

    # --- 处理已存在的目录 ---
    if os.path.isdir(output_dir) and os.listdir(output_dir):
        if force:
            print(f"[INFO] 覆盖已有向量库：{output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"[INFO] 向量库已存在，跳过：{output_dir}")
            print("[INFO] 如需重建，请使用 --force 参数。")
            return

    os.makedirs(output_dir, exist_ok=True)

    source = records[0].get("source", "unknown")

    # --- 批量 embedding ---
    texts = [r["text"] for r in records]
    total = len(texts)

    all_vectors: List[np.ndarray] = []
    for start in tqdm(range(0, total, batch_size), desc=f"Embedding {source}"):
        batch = texts[start : start + batch_size]
        vectors = embedding_fn(batch)
        all_vectors.append(np.asarray(vectors, dtype="float32"))

    # 合并所有批次
    vectors = np.concatenate(all_vectors, axis=0)

    if vectors.ndim != 2:
        raise ValueError(f"embedding_fn 返回了非二维数组：shape={vectors.shape}")

    dim = int(vectors.shape[1])

    # --- L2 归一化 + FAISS IndexFlatIP ---
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # --- 保存 ---
    index_path = os.path.join(output_dir, "index.faiss")
    faiss.write_index(index, index_path)
    print(f"[INFO] Saved index: {index_path}  ({index.ntotal} vectors, dim={dim})")

    meta_path = os.path.join(output_dir, "meta.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved meta: {meta_path}  ({len(records)} records)")

    state = get_embedding_state()
    config = {
        "source": source,
        "num_records": len(records),
        "dim": dim,
        "embedding_backend": state.get("backend", "unknown"),
        "embedding_model": str(state.get("model", "unknown")),
        "created_at": datetime.now().isoformat(),
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved config: {config_path}")

    print(
        f"[DONE] Built store '{source}': {len(records)} records, dim={dim}, "
        f"backend={config['embedding_backend']}, model={config['embedding_model']}"
    )
