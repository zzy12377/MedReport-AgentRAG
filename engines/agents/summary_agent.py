# -*- coding: utf-8 -*-
"""Summary agent skeleton."""

from __future__ import annotations

from typing import Dict, Iterable


def summarize_agent_outputs(agent_outputs: Iterable[Dict[str, object]], critique: Dict[str, object]) -> Dict[str, object]:
    outputs = list(agent_outputs)
    risk_order = {"low": 0, "medium": 1, "high": 2}
    max_risk = "low"
    for output in outputs:
        risk = str(output.get("risk_level", "low"))
        if risk_order.get(risk, 0) > risk_order.get(max_risk, 0):
            max_risk = risk
    return {
        "overall_risk": max_risk,
        "agent_outputs": outputs,
        "critique": critique,
        "recommendations": ["Review abnormal indicators with a qualified clinician."],
    }

