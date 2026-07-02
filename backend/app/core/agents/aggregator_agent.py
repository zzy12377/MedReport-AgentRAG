# -*- coding: utf-8 -*-
"""Aggregator adapter."""

from __future__ import annotations

from typing import Any, Dict, List


class AggregatorAgent:
    def aggregate(
        self,
        opinions: List[Dict[str, Any]],
        critique: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        diagnoses = []
        for row in retrieved_cases[:3]:
            diagnosis = str(row.get("diagnosis") or "").strip()
            if diagnosis and diagnosis not in diagnoses:
                diagnoses.append(diagnosis)
        if not diagnoses:
            diagnoses = ["general medical risk"]
        return {
            "overall_risk": "medium" if kg_evidence or retrieved_cases else "unknown",
            "possible_diagnoses": diagnoses,
            "summary_markdown": "\n".join(
                [
                    "## Diagnosis Summary",
                    "",
                    f"- Possible diagnoses: {', '.join(diagnoses)}",
                    f"- Retrieved cases: {len(retrieved_cases)}",
                    f"- KG evidence: {len(kg_evidence)}",
                    "- Safety note: this result is for course demonstration and reference only.",
                ]
            ),
            "followup_questions": [
                "Please confirm symptom onset time and duration.",
                "Please confirm whether the abnormal indicators were repeated or fasting values.",
            ],
        }
