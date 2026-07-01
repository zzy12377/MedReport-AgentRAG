# -*- coding: utf-8 -*-
"""Thin embedding wrapper around the existing MedRAG embedding backend."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from embedding_backend import (
    get_embedding_state,
    local_sentence_transformer_embeddings,
    set_embedding_state,
)


class EmbeddingEngine:
    """Generate embeddings with SiliconFlow first and local fallback."""

    def __init__(self, force_local: bool = False, model: Optional[str] = None):
        self.force_local = force_local
        self.model = model

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        text_list = [str(t or "empty medical text") for t in texts]
        if not text_list:
            return np.empty((0, 0), dtype="float32")

        if self.force_local:
            return local_sentence_transformer_embeddings(text_list, reason="forced local embedding engine")

        try:
            from authentication import api_key, base_url, embedding_model
            import faiss
            import openai

            model_name = self.model or embedding_model
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            vectors: List[List[float]] = []
            for text in text_list:
                response = client.embeddings.create(input=text, model=model_name)
                vectors.append(response.data[0].embedding)
            arr = np.asarray(vectors, dtype="float32")
            faiss.normalize_L2(arr)
            set_embedding_state(backend="remote", model=model_name, dim=int(arr.shape[1]))
            return arr
        except Exception as exc:
            return local_sentence_transformer_embeddings(
                text_list,
                reason=f"remote embedding failed in EmbeddingEngine: {exc}",
            )

    def embed_query(self, text: str) -> np.ndarray:
        vectors = self.embed_texts([text])
        if vectors.size == 0:
            return np.empty((0,), dtype="float32")
        return vectors[0]

    @property
    def state(self) -> dict:
        return get_embedding_state()


def embed_texts(texts: Iterable[str], force_local: bool = False) -> np.ndarray:
    return EmbeddingEngine(force_local=force_local).embed_texts(texts)

