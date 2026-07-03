# -*- coding: utf-8 -*-
"""KG subgraph retrieval adapter."""

from __future__ import annotations

from typing import Any, Dict, List

from backend.app.config.settings import settings
from engines.kg.chinese_kg import ChineseMedicalKGRetriever
from engines.kg.kg_extractor import DDXPlusKGRetriever


class KGSubgraphRetriever:
    def __init__(self) -> None:
        self._retriever: DDXPlusKGRetriever | None = None
        self._chinese_retriever: ChineseMedicalKGRetriever | None = None

    def _get_retriever(self) -> DDXPlusKGRetriever:
        if self._retriever is None:
            self._retriever = DDXPlusKGRetriever(
                kg_path=settings.preferred_kg_path(),
                vector_base_dir=settings.vector_base_dir,
                use_vector=True,
            )
        return self._retriever

    def _get_chinese_retriever(self) -> ChineseMedicalKGRetriever:
        if self._chinese_retriever is None:
            self._chinese_retriever = ChineseMedicalKGRetriever(settings.chinese_kg_path)
        return self._chinese_retriever

    def retrieve(self, features: Dict[str, Any], raw_text: str = "", top_k: int | None = None) -> List[Dict[str, Any]]:
        indicators = features.get("indicators", []) if isinstance(features, dict) else []
        query_text = raw_text or str(features.get("symptoms", "")) if isinstance(features, dict) else raw_text
        limit = top_k or settings.default_kg_top_k
        chinese_rows = self._get_chinese_retriever().retrieve(indicators, top_k=limit)
        ddx_rows = self._get_retriever().retrieve(
            query_text=query_text,
            entities=indicators,
            top_k=limit,
        )
        merged = []
        seen = set()
        for row in chinese_rows + ddx_rows:
            key = (
                str(row.get("source", "")),
                str(row.get("head", "")),
                str(row.get("relation", "")),
                str(row.get("tail", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(row)
            if len(merged) >= limit:
                break
        return merged
