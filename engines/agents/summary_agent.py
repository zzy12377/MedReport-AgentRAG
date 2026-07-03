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
        "recommendations": _recommendations(outputs, critique),
    }


def _recommendations(outputs: list[Dict[str, object]], critique: Dict[str, object]) -> list[str]:
    rows = []
    for output in outputs:
        suggestion = str(output.get("suggestion") or "").strip()
        if suggestion and suggestion not in rows and output.get("risk_level") in {"medium", "high"}:
            rows.append(suggestion)
    if critique.get("conflicts"):
        rows.append("存在跨专科风险聚集，建议由医生结合病史和复查结果综合判断。")
    rows.append("本结果仅用于课程演示和辅助参考，不能替代医生诊断。")
    return rows
