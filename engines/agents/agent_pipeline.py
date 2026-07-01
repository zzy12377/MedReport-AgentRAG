# -*- coding: utf-8 -*-
"""Phase-1 multi-agent pipeline wrapper."""

from __future__ import annotations

from typing import Dict, Iterable, List

from engines.agents.critique_engine import critique_agent_outputs
from engines.agents.specialist_agent import SpecialistAgent
from engines.agents.summary_agent import summarize_agent_outputs


def run_agent_pipeline(entities: Iterable[Dict[str, object]], retrieved_cases: List[Dict[str, object]] | None = None) -> Dict[str, object]:
    entity_list = list(entities)
    outputs = [
        SpecialistAgent("cardiovascular").analyze(entity_list, retrieved_cases),
        SpecialistAgent("liver").analyze(entity_list, retrieved_cases),
        SpecialistAgent("endocrine").analyze(entity_list, retrieved_cases),
    ]
    critique = critique_agent_outputs(outputs)
    return summarize_agent_outputs(outputs, critique)

