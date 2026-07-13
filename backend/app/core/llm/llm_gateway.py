# -*- coding: utf-8 -*-
"""LLM gateway adapter."""

from __future__ import annotations

from engines.llm.llm_gateway import LLMGateway as _LLMGateway


class LLMGateway(_LLMGateway):
    _instance: "LLMGateway | None" = None

    @classmethod
    def get_instance(cls, mock: bool = False) -> "LLMGateway":
        if cls._instance is None or mock:
            cls._instance = cls(mock=mock)
        return cls._instance
