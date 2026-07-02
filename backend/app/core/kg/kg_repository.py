# -*- coding: utf-8 -*-
"""Knowledge graph repository adapter."""

from __future__ import annotations

from typing import Any, Dict, List

from backend.app.config.settings import settings
from engines.kg.kg_extractor import load_kg_triples


class KGRepository:
    _instance: "KGRepository | None" = None

    def __init__(self) -> None:
        self.triples: List[Dict[str, Any]] = []
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "KGRepository":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self.triples = load_kg_triples(settings.preferred_kg_path())

    def all(self) -> List[Dict[str, Any]]:
        self.load()
        return list(self.triples)
