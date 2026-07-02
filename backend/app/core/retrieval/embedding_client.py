# -*- coding: utf-8 -*-
"""Embedding adapter used by the document-aligned backend."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from engines.retrieval.embedding_engine import EmbeddingEngine


class EmbeddingClient:
    def __init__(self, force_local: bool = True, batch_size: int = 32) -> None:
        self.engine = EmbeddingEngine(force_local=force_local, batch_size=batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self.engine.embed_query(text)

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        return self.engine.embed_texts(list(texts))
