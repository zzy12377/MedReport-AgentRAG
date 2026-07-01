# -*- coding: utf-8 -*-
"""Specialist agent skeletons for cardiovascular, liver and endocrine review."""

from __future__ import annotations

from typing import Dict, Iterable, List


SPECIALTIES = {
    "cardiovascular": {"signals": {"TC", "TG", "LDL-C", "HDL-C", "收缩压", "舒张压"}},
    "liver": {"signals": {"ALT", "AST", "GGT", "TBIL"}},
    "endocrine": {"signals": {"GLU", "HbA1c", "尿酸"}},
}


class SpecialistAgent:
    def __init__(self, specialty: str):
        self.specialty = specialty
        self.signals = SPECIALTIES.get(specialty, {}).get("signals", set())

    def analyze(self, entities: Iterable[Dict[str, object]], retrieved_cases: List[Dict[str, object]] | None = None) -> Dict[str, object]:
        matched = [e for e in entities if e.get("name") in self.signals]
        abnormal = [e for e in matched if e.get("is_abnormal")]
        risk = "high" if len(abnormal) >= 2 else "medium" if abnormal else "low"
        confidence = 0.75 if abnormal else 0.45
        return {
            "specialty": self.specialty,
            "risk_level": risk,
            "confidence": confidence,
            "evidence": matched,
            "retrieved_case_count": len(retrieved_cases or []),
        }

