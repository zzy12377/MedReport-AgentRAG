# -*- coding: utf-8 -*-
"""Smoke test for detailed detection-report sections."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.services.pipeline import DiagnosisPipeline


def main() -> int:
    entities = [
        {"name": "GLU", "value": 7.2, "unit": "mmol/L", "is_abnormal": True},
        {"name": "LDL-C", "value": 4.1, "unit": "mmol/L", "is_abnormal": True},
    ]
    retrieved_cases = [
        {"case_id": "case-001", "diagnosis": "Diabetes", "similarity": 0.81, "raw_text": "similar diabetes case"}
    ]
    kg_evidence = [
        {
            "head": "Diabetes",
            "relation": "has symptomatology",
            "tail": "Polyuria and polydipsia",
            "score": 4.5,
            "relation_category": "symptom",
        },
        {
            "head": "Diabetes",
            "relation": "has risk factor",
            "tail": "Elevated glucose",
            "score": 3.0,
            "relation_category": "risk_factor",
        },
    ]
    agent_outputs = [{"specialty": "endocrine", "risk_level": "high", "confidence": 0.75}]
    conclusion = DiagnosisPipeline._detection_conclusion(
        text="GLU 7.2 mmol/L LDL-C 4.1 mmol/L",
        prediction="Diabetes",
        possible_diagnoses=["Diabetes"],
        retrieved_cases=retrieved_cases,
        entities=entities,
        overall_risk="high",
    )
    kg_symptoms = DiagnosisPipeline._kg_disease_symptoms("Diabetes", kg_evidence)
    rates = DiagnosisPipeline._baseline_match_rates(
        text="GLU 7.2 mmol/L LDL-C 4.1 mmol/L",
        prediction="Diabetes",
        primary_diagnosis="Diabetes",
        retrieved_cases=retrieved_cases,
        kg_evidence=kg_evidence,
        agent_outputs=agent_outputs,
        entities=entities,
    )
    markdown = DiagnosisPipeline._summary_markdown(
        detection_conclusion=conclusion,
        overall_risk="high",
        possible_diagnoses=["Diabetes"],
        baseline_match_rates=rates,
        kg_disease_symptoms=kg_symptoms,
        retrieved_cases=retrieved_cases,
        kg_evidence=kg_evidence,
        agent_outputs=agent_outputs,
        llm_response="Mock summary",
    )
    assert conclusion["primary_diagnosis"] == "血糖异常 / 糖代谢风险", conclusion
    bp_conclusion = DiagnosisPipeline._detection_conclusion(
        text="姓名 张三\n血压 150/95 mmHg",
        prediction="**Diagnosis-Oriented Summary:** The patient presents with hypertension",
        possible_diagnoses=["**Diagnosis-Oriented Summary:** The patient presents with hypertension"],
        retrieved_cases=[],
        entities=[
            {"name": "收缩压", "value": 150, "unit": "mmHg", "is_abnormal": True},
            {"name": "舒张压", "value": 95, "unit": "mmHg", "is_abnormal": True},
        ],
        overall_risk="high",
    )
    assert bp_conclusion["primary_diagnosis"] == "血压升高 / 高血压风险", bp_conclusion
    no_abnormal = DiagnosisPipeline._detection_conclusion(
        text="结论 各项指标基本正常，建议保持规律作息。",
        prediction="Mock normal",
        possible_diagnoses=[],
        retrieved_cases=[],
        entities=[],
        overall_risk="unknown",
    )
    assert no_abnormal["primary_diagnosis"] == "未发现明确异常指标", no_abnormal
    assert "conclusion" in no_abnormal["basis"], no_abnormal
    assert kg_symptoms, kg_symptoms
    assert [row["mode"] for row in rates] == ["B0", "B1", "B2"], rates
    for heading in ["一、检测结论", "二、知识图谱对应疾病症状", "三、B0 / B1 / B2 匹配率"]:
        assert heading in markdown, markdown
    print("[OK] detailed detection report smoke test passed.")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
