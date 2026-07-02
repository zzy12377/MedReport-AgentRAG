# -*- coding: utf-8 -*-
"""FAISS index adapter."""

from __future__ import annotations

from typing import Any, Dict, List

from engines.retrieval.faiss_retriever import FaissCaseRetriever


class FaissIndex:
    def __init__(self, train_dir: str = "./dataset/df/train", top_k: int = 5) -> None:
        self.retriever = FaissCaseRetriever(train_dir=train_dir, top_k=top_k, force_local=True)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return self.retriever.retrieve_similar_cases(query, top_k=k)
