# -*- coding: utf-8 -*-
"""Optional multi-source vector retrieval for B1/B2 baselines.

The default B1/B2 path still uses the dedicated DDXPlus FAISS case retriever.
This wrapper is only used when a caller explicitly passes vector sources, for
example ``--vector-sources all`` or ``--vector-sources ddxplus_cases ddxplus_kg``.
It reuses the vector_store/ indexes built by scripts/build_vector_stores.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from vector_store.registry import MultiVectorRetriever
from vector_store.utils import create_embedding_fn, create_query_embedding_fn


def normalize_vector_sources(sources: Optional[Iterable[str]]) -> Optional[List[str]]:
    if sources is None:
        return None
    cleaned = [str(source).strip() for source in sources if str(source).strip()]
    if not cleaned:
        return None
    if any(source.lower() == "all" for source in cleaned):
        return None
    return cleaned


def standardize_vector_result(row: Dict[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return {
        "case_id": str(row.get("id") or metadata.get("participant_no") or ""),
        "source": str(row.get("source") or ""),
        "title": str(row.get("title") or ""),
        "diagnosis": str(row.get("diagnosis") or ""),
        "similarity": float(row.get("score", 0.0)),
        "raw_text": str(row.get("text") or ""),
        "metadata": metadata,
    }


class MultiSourceRetriever:
    """Thin adapter around vector_store.MultiVectorRetriever."""

    def __init__(self, base_dir: str = "./vector_db", force_local: bool = True):
        self.base_dir = os.path.normpath(base_dir)
        self.force_local = force_local
        self.registry = MultiVectorRetriever(base_dir=self.base_dir)
        embedding_fn = create_embedding_fn(force_local=force_local)
        self.query_embedding_fn = create_query_embedding_fn(embedding_fn)

    @property
    def sources(self) -> List[str]:
        return self.registry.sources

    def retrieve(
        self,
        query_text: str,
        sources: Optional[Iterable[str]] = None,
        top_k: int = 5,
        top_k_per_source: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        active_sources = normalize_vector_sources(sources)
        if not self.sources:
            return []
        per_source = int(top_k_per_source or top_k)
        results = self.registry.search(
            query=query_text,
            embedding_fn=self.query_embedding_fn,
            sources=active_sources,
            top_k_per_source=per_source,
            final_top_k=int(top_k),
        )
        return [standardize_vector_result(row) for row in results]

