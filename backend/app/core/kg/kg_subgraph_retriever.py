# -*- coding: utf-8 -*-
"""KG subgraph retrieval adapter."""

from __future__ import annotations

from typing import Any, Dict, List

from backend.app.config.settings import settings
from engines.kg.kg_extractor import DDXPlusKGRetriever


class KGSubgraphRetriever:
    def __init__(self) -> None:
        self._retriever: DDXPlusKGRetriever | None = None

    def _get_retriever(self) -> DDXPlusKGRetriever:
        if self._retriever is None:
            self._retriever = DDXPlusKGRetriever(
                kg_path=settings.preferred_kg_path(),
                vector_base_dir=settings.vector_base_dir,
                use_vector=not settings.use_zh_data,
            )
        return self._retriever

    def retrieve(self, features: Dict[str, Any], raw_text: str = "", top_k: int | None = None) -> List[Dict[str, Any]]:
        indicators = features.get("indicators", []) if isinstance(features, dict) else []
        query_text = raw_text or str(features.get("symptoms", "")) if isinstance(features, dict) else raw_text
        return self._get_retriever().retrieve(
            query_text=query_text,
            entities=indicators,
            top_k=top_k or settings.default_kg_top_k,
        )
