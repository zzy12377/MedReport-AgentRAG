# -*- coding: utf-8 -*-
"""Critique and confidence calibration for rule-based specialist agents."""

from __future__ import annotations

from typing import Dict, Iterable, List


def critique_agent_outputs(agent_outputs: Iterable[Dict[str, object]]) -> Dict[str, object]:
    outputs = list(agent_outputs)
    low_confidence = [o.get("specialty") for o in outputs if float(o.get("confidence", 0.0)) < 0.6]
    high_without_kg = [
        o.get("specialty")
        for o in outputs
        if o.get("risk_level") == "high" and int(o.get("kg_evidence_count", 0) or 0) == 0
    ]
    active = [o for o in outputs if o.get("risk_level") in {"medium", "high"}]
    conflicts = []
    if len(active) >= 3:
        conflicts.append(
            {
                "type": "multi_system_risk",
                "message": "多个专科同时存在中高风险，需优先考虑代谢综合征或系统性疾病背景。",
                "agents": [o.get("specialty") for o in active],
            }
        )
    if any(o.get("specialty") == "cardiovascular" and o.get("risk_level") == "high" for o in outputs) and any(
        o.get("specialty") == "endocrine" and o.get("risk_level") == "high" for o in outputs
    ):
        conflicts.append(
            {
                "type": "cardio_metabolic_cluster",
                "message": "心血管与内分泌代谢风险同时较高，应关注血压、血脂、血糖和体重的综合管理。",
            }
        )
    return {
        "conflicts": conflicts,
        "low_confidence_agents": low_confidence,
        "high_risk_without_kg_support": high_without_kg,
        "calibration_note": "Confidence is calibrated by abnormal indicators, KG support and retrieval availability.",
    }
