# -*- coding: utf-8 -*-
"""Critique adapter."""

from __future__ import annotations

from typing import Any, Dict, List

from engines.agents.critique_engine import critique_agent_outputs


class CritiqueEngine:
    def run(self, opinions: List[Dict[str, Any]], kg_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        critique = critique_agent_outputs(opinions)
        if isinstance(critique, dict):
            critique.setdefault("kg_evidence_count", len(kg_evidence))
            return critique
        return {"summary": str(critique), "kg_evidence_count": len(kg_evidence)}
