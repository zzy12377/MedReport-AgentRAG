# -*- coding: utf-8 -*-
"""Structured diagnosis result schemas."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class AgentOpinion(BaseModel):
    agent_name: str
    risk_level: str = "unknown"
    diagnosis: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    suggestion: str = ""


class CritiqueReport(BaseModel):
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    kg_consistency_issues: List[Dict[str, Any]] = Field(default_factory=list)
    low_confidence_flags: List[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    task_id: str
    overall_risk: str
    possible_diagnoses: List[str]
    detection_conclusion: Dict[str, Any] = Field(default_factory=dict)
    kg_disease_symptoms: List[Dict[str, Any]] = Field(default_factory=list)
    baseline_match_rates: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_cases: List[Dict[str, Any]] = Field(default_factory=list)
    kg_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    agent_opinions: List[Dict[str, Any]] = Field(default_factory=list)
    critique: Dict[str, Any] = Field(default_factory=dict)
    summary_markdown: str
    followup_questions: List[str] = Field(default_factory=list)
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    raw_baseline_result: Dict[str, Any] = Field(default_factory=dict)
    safety_note: str = "本结果仅用于课程演示和辅助参考，不能替代医生诊断。"
