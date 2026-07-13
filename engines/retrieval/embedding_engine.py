# -*- coding: utf-8 -*-
"""Embedding wrapper with batching and persistent DDXPlus cache."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from embedding_backend import (
    get_embedding_state,
    local_sentence_transformer_embeddings,
    set_embedding_state,
)


DEFAULT_EMBEDDINGS_PATH = "./storage/embeddings/ddxplus_cases.npy"
DEFAULT_METADATA_PATH = "./storage/embeddings/ddxplus_cases_metadata.jsonl"


def text_hash(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="ignore")).hexdigest()


def _chunks(items: List[str], size: int) -> Iterable[List[str]]:
    size = max(1, int(size or 32))
    for start in range(0, len(items), size):
        yield items[start : start + size]


class EmbeddingEngine:
    """Generate embeddings with SiliconFlow first and local fallback."""

    def __init__(self, force_local: bool = False, model: Optional[str] = None, batch_size: int = 32):
        self.force_local = force_local
        self.model = model
        self.batch_size = max(1, int(batch_size or 32))

    def _remote_embed_batch(self, texts: List[str]) -> np.ndarray:
        from authentication import api_key, base_url, embedding_model
        import faiss
        import openai

        model_name = self.model or embedding_model
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        vectors: List[List[float]] = []
        for text in texts:
            response = client.embeddings.create(input=text, model=model_name)
            vectors.append(response.data[0].embedding)
        arr = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(arr)
        set_embedding_state(backend="remote", model=model_name, dim=int(arr.shape[1]))
        return arr

    def embed_texts(self, texts: Iterable[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Embed texts in batches and never let remote failures crash the caller."""
        text_list = [str(t or "empty medical text") for t in texts]
        if not text_list:
            return np.empty((0, 0), dtype="float32")

        active_batch_size = max(1, int(batch_size or self.batch_size))
        vectors: List[np.ndarray] = []
        total_batches = (len(text_list) + active_batch_size - 1) // active_batch_size

        for batch_no, batch in enumerate(_chunks(text_list, active_batch_size), start=1):
            print(f"[INFO] Embedding batch {batch_no}/{total_batches}, size={len(batch)}")
            if self.force_local:
                batch_vectors = local_sentence_transformer_embeddings(
                    batch,
                    reason="forced local embedding engine",
                )
            else:
                try:
                    batch_vectors = self._remote_embed_batch(batch)
                except Exception as exc:
                    batch_vectors = local_sentence_transformer_embeddings(
                        batch,
                        reason=f"remote embedding failed in EmbeddingEngine: {exc}",
                    )
            vectors.append(np.asarray(batch_vectors, dtype="float32"))

        return np.concatenate(vectors, axis=0)

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embed_texts([text], batch_size=1)
        if vectors.size == 0:
            return np.empty((0,), dtype="float32")
        return vectors[0]

    def load_cached_embeddings(
        self,
        records: List[Dict[str, Any]],
        embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
    ) -> Optional[np.ndarray]:
        if not (os.path.exists(embeddings_path) and os.path.exists(metadata_path)):
            return None
        try:
            vectors = np.load(embeddings_path).astype("float32")
            metadata = load_embedding_metadata(metadata_path)
        except Exception as exc:
            print(f"[WARN] Failed to load embedding cache: {exc}")
            return None

        if vectors.ndim != 2 or vectors.shape[0] != len(records) or len(metadata) != len(records):
            print("[WARN] Embedding cache count mismatch; rebuilding embeddings.")
            return None

        for record, meta in zip(records, metadata):
            if str(record.get("case_id")) != str(meta.get("case_id")):
                print("[WARN] Embedding cache case_id mismatch; rebuilding embeddings.")
                return None
            if text_hash(str(record.get("raw_text", ""))) != str(meta.get("text_sha1")):
                print("[WARN] Embedding cache text hash mismatch; rebuilding embeddings.")
                return None

        print(f"[INFO] Loaded cached embeddings: {embeddings_path}, shape={vectors.shape}")
        return vectors

    def embed_records_with_cache(
        self,
        records: List[Dict[str, Any]],
        embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
        metadata_path: str = DEFAULT_METADATA_PATH,
        batch_size: Optional[int] = None,
        force_rebuild: bool = False,
    ) -> np.ndarray:
        """Embed case records and persist a reusable DDXPlus cache."""
        if not force_rebuild:
            cached = self.load_cached_embeddings(records, embeddings_path, metadata_path)
            if cached is not None:
                return cached

        texts = [str(record.get("raw_text", "")) for record in records]
        vectors = self.embed_texts(texts, batch_size=batch_size)
        save_embedding_cache(records, vectors, embeddings_path, metadata_path)
        return vectors

    @property
    def state(self) -> dict:
        return get_embedding_state()


def load_embedding_metadata(metadata_path: str = DEFAULT_METADATA_PATH) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(metadata_path):
        return rows
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_embedding_cache(
    records: List[Dict[str, Any]],
    vectors: np.ndarray,
    embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> None:
    os.makedirs(os.path.dirname(embeddings_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metadata_path) or ".", exist_ok=True)
    np.save(embeddings_path, np.asarray(vectors, dtype="float32"))

    state = get_embedding_state()
    dim = int(vectors.shape[1]) if vectors.ndim == 2 else None
    with open(metadata_path, "w", encoding="utf-8") as f:
        for idx, record in enumerate(records):
            row = {
                "row_index": idx,
                "case_id": str(record.get("case_id", "")),
                "diagnosis": str(record.get("diagnosis", "")),
                "source_path": str(record.get("source_path", "")),
                "text_sha1": text_hash(str(record.get("raw_text", ""))),
                "embedding_backend": state.get("backend"),
                "embedding_model": state.get("model"),
                "dim": dim,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[INFO] Saved embeddings: {embeddings_path}, shape={vectors.shape}")
    print(f"[INFO] Saved embedding metadata: {metadata_path}")


def embed_texts(texts: Iterable[str], force_local: bool = False, batch_size: int = 32) -> np.ndarray:
    return EmbeddingEngine(force_local=force_local, batch_size=batch_size).embed_texts(texts)

