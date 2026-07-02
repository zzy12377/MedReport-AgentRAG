# -*- coding: utf-8 -*-
"""Base agent placeholder compatible with the document structure."""

from __future__ import annotations

from typing import Any, Dict, List


class BaseAgent:
    agent_name = "base"

    async def run(
        self,
        features: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "risk_level": "unknown",
            "diagnosis": [],
            "evidence": [],
            "confidence": 0.0,
            "suggestion": "",
        }
