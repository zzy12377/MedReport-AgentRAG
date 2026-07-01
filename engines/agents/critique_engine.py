# -*- coding: utf-8 -*-
"""Critique and confidence calibration skeleton."""

from __future__ import annotations

from typing import Dict, Iterable, List


def critique_agent_outputs(agent_outputs: Iterable[Dict[str, object]]) -> Dict[str, object]:
    outputs = list(agent_outputs)
    low_confidence = [o.get("specialty") for o in outputs if float(o.get("confidence", 0.0)) < 0.6]
    return {
        "conflicts": [],
        "low_confidence_agents": low_confidence,
        "calibration_note": "Phase-1 rule critique; KG consistency checks are planned for phase 2.",
    }

