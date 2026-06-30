# -*- coding: utf-8 -*-
"""
vector_store/registry.py

MultiVectorRetriever: 多向量库联合检索。

用法：
    retriever = MultiVectorRetriever(base_dir="./vector_db")
    results = retriever.search(
        query="fever and cough",
        embedding_fn=query_embedding_fn,
        sources=["ddxplus_cases", "pmc_patients"],
        top_k_per_source=5,
        final_top_k=10,
    )
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional

import numpy as np

from vector_store.retriever import search_store


class MultiVectorRetriever:
    """
    管理 vector_db/ 下所有子库，支持指定来源的联邦检索。
    """

    def __init__(self, base_dir: str = "./vector_db"):
        """
        扫描 base_dir 下的所有子目录，找到包含 index.faiss 的目录。

        Args:
            base_dir: 向量库根目录。
        """
        base_dir = os.path.normpath(base_dir)
        self.base_dir = base_dir
        self._stores: Dict[str, str] = {}

        if not os.path.isdir(base_dir):
            print(f"[WARN] 向量库根目录不存在：{base_dir}")
            print("[INFO] 请先运行 python scripts/build_vector_stores.py 构建向量库。")
            return

        for entry in sorted(os.listdir(base_dir)):
            entry_path = os.path.join(base_dir, entry)
            if not os.path.isdir(entry_path):
                continue

            index_path = os.path.join(entry_path, "index.faiss")
            meta_path = os.path.join(entry_path, "meta.jsonl")

            if os.path.exists(index_path) and os.path.exists(meta_path):
                self._stores[entry] = entry_path
                # 读取 config 获取记录数
                config_path = os.path.join(entry_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f:
                        config = json.load(f)
                    n = config.get("num_records", "?")
                    dim = config.get("dim", "?")
                    backend = config.get("embedding_backend", "?")
                    print(f"[INFO] Found store: {entry:25s}  {n} records, dim={dim}, backend={backend}")
                else:
                    print(f"[INFO] Found store: {entry:25s}  (no config.json)")

        if not self._stores:
            print(f"[WARN] 在 {base_dir} 下未找到任何向量库。")

    @property
    def sources(self) -> List[str]:
        """所有可用向量库名称（排序）。"""
        return sorted(self._stores.keys())

    def search(
        self,
        query: str,
        embedding_fn: Callable[[str], np.ndarray],
        sources: Optional[List[str]] = None,
        top_k_per_source: int = 5,
        final_top_k: int = 10,
    ) -> List[dict]:
        """
        在多个向量库中检索并合并结果。

        Args:
            query: 查询文本。
            embedding_fn: 查询 embedding 函数，fn(str) -> np.ndarray。
            sources: 要检索的向量库列表。None 表示检索所有可用库。
            top_k_per_source: 每个库返回的结果数。
            final_top_k: 合并后最终返回的结果数。

        Returns:
            List[dict]，按 score 降序排列，最多 final_top_k 条。
        """
        if sources is None:
            active_sources = self.sources
        else:
            active_sources = []
            for s in sources:
                if s in self._stores:
                    active_sources.append(s)
                else:
                    print(f"[WARN] 向量库不存在，跳过：{s}。可用：{self.sources}")

        if not active_sources:
            print("[WARN] 没有可检索的向量库。")
            return []

        all_results: List[dict] = []

        for source_name in active_sources:
            store_dir = self._stores[source_name]
            try:
                results = search_store(
                    query=query,
                    store_dir=store_dir,
                    top_k=top_k_per_source,
                    embedding_fn=embedding_fn,
                )
                all_results.extend(results)
            except Exception as e:
                print(f"[WARN] 检索 {source_name} 失败：{e}")
                continue

        if not all_results:
            return []

        # 按 score 降序排列
        all_results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

        return all_results[:final_top_k]
