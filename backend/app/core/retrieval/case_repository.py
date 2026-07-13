# -*- coding: utf-8 -*-
"""Runtime case repository.

This adapter prefers the optional multi-vector stores under ``vector_db``. If
they are not available it falls back to the full DDXPlus FAISS case index.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

from backend.app.config.settings import settings
from engines.retrieval.faiss_retriever import FaissCaseRetriever
from engines.retrieval.multi_source_retriever import MultiSourceRetriever, normalize_vector_sources


class CaseRepository:
    _instance: "CaseRepository | None" = None

    def __init__(self) -> None:
        self._multi: MultiSourceRetriever | None = None
        self._external_multi: MultiSourceRetriever | None = None
        self._faiss: FaissCaseRetriever | None = None
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "CaseRepository":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if settings.use_multi_vector and not settings.use_zh_data and os.path.isdir(settings.vector_base_dir):
            try:
                multi = MultiSourceRetriever(base_dir=settings.vector_base_dir, force_local=True)
                if multi.sources:
                    self._multi = multi
            except Exception as exc:
                print(f"[WARN] Multi-vector repository unavailable: {exc}")
        if settings.use_external_vector and os.path.isdir(settings.external_vector_base_dir):
            try:
                external = MultiSourceRetriever(base_dir=settings.external_vector_base_dir, force_local=True)
                if external.sources:
                    self._external_multi = external
            except Exception as exc:
                print(f"[WARN] External zh vector repository unavailable: {exc}")
        if self._multi is None or settings.use_zh_data:
            try:
                self._faiss = FaissCaseRetriever(
                    train_dir=settings.preferred_train_dir(),
                    index_path=settings.case_zh_index_path if settings.use_zh_data else settings.case_index_path,
                    metadata_path=settings.case_zh_index_metadata_path if settings.use_zh_data else settings.case_index_metadata_path,
                    embeddings_path=settings.case_zh_embeddings_path if settings.use_zh_data else settings.case_embeddings_path,
                    embedding_metadata_path=(
                        settings.case_zh_embedding_metadata_path
                        if settings.use_zh_data
                        else settings.case_embedding_metadata_path
                    ),
                    top_k=settings.default_top_k,
                    force_local=True,
                )
            except Exception as exc:
                print(f"[WARN] DDXPlus FAISS repository unavailable: {exc}")

    def search(
        self,
        query: str,
        k: int = 5,
        sources: Optional[Iterable[str]] = None,
        top_k_per_source: int | None = None,
    ) -> List[Dict[str, Any]]:
        self.load()
        if settings.use_zh_data:
            results: List[Dict[str, Any]] = []
            if self._faiss is not None:
                results.extend(self._faiss.retrieve_similar_cases(query, top_k=k))
            if self._external_multi is not None:
                active_sources = normalize_vector_sources(
                    sources or settings.preferred_external_vector_sources()
                )
                results.extend(
                    self._external_multi.retrieve(
                        query,
                        sources=active_sources,
                        top_k=max(k, settings.default_top_k_per_source * max(1, len(self._external_multi.sources))),
                        top_k_per_source=top_k_per_source or settings.default_top_k_per_source,
                    )
                )
            return _merge_results(results, top_k=k)
        if self._multi is not None:
            active_sources = normalize_vector_sources(sources or ["all"])
            return self._multi.retrieve(
                query,
                sources=active_sources,
                top_k=k,
                top_k_per_source=top_k_per_source or settings.default_top_k_per_source,
            )
        if self._faiss is not None:
            return self._faiss.retrieve_similar_cases(query, top_k=k)
        return []


def _merge_results(rows: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id") or row.get("id") or row.get("title") or "")
        source = str(row.get("source") or "")
        key = f"{source}:{case_id}"
        if not key.strip(":"):
            key = str(len(merged))
        current = merged.get(key)
        if current is None or float(row.get("similarity", 0.0) or 0.0) > float(current.get("similarity", 0.0) or 0.0):
            merged[key] = row
    results = list(merged.values())
    results.sort(key=lambda item: float(item.get("similarity", 0.0) or 0.0), reverse=True)
    return results[: max(0, int(top_k or 0))]
