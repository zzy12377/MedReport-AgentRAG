# -*- coding: utf-8 -*-
"""Optional LLM-based MDT multi-agent workflow.

This module adapts the external Multi_Agent.py idea to the current FastAPI
pipeline. It takes already-normalized runtime evidence instead of legacy
participant_no / KG_Retrieve inputs, and uses the shared LLMGateway.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
from typing import Any, Dict, List


SPECIALISTS = [
    (
        "cardiovascular",
        "心血管内科",
        "评估血压、心率、血脂、胸痛、呼吸困难、心衰、冠心病、心律失常等心血管风险。",
    ),
    (
        "digestive_liver",
        "消化与肝胆内科",
        "评估 ALT、AST、GGT、胆红素、腹痛、消化道症状、肝胆胰相关风险。",
    ),
    (
        "endocrine_metabolic",
        "内分泌与代谢科",
        "评估血糖、HbA1c、BMI、血脂、尿酸、甲状腺和代谢综合征相关风险。",
    ),
]


SELF_CRITIQUE_PROMPT = """你是一位临床医学专家 Agent。请针对你刚刚给出的初步评估，结合患者原始文本、结构化指标、相似病例和知识图谱证据进行 Self-Critique。
要求：
1. 删除并标注任何不来自输入证据的推断。
2. 重新核对支持证据和排除证据。
3. 修正疾病名称、风险等级和置信度。
4. 只返回 JSON，不要 Markdown。

JSON 格式：
{
  "specialty": "专科名称",
  "revised_suspected_disease": "修正后的疑似疾病或风险方向",
  "revised_risk_level": "low/medium/high",
  "revised_confidence": 0.0,
  "revised_supporting_evidence": ["证据1"],
  "revised_exclusion_evidence": ["排除证据1"],
  "self_critique_log": "自审说明"
}
"""


CRITIQUE_PROMPT = """你是中立的临床证据审计与置信度校准引擎。请审计三位专科 Agent 的修正意见。
要求：
1. 检查 supporting evidence 是否真实来自输入。
2. 检查是否遗漏关键阴性证据。
3. 检查诊断是否和知识图谱/相似病例证据一致。
4. 重新校准置信度。
5. 只返回 JSON，不要 Markdown。

JSON 格式：
{
  "evidence_verification": {
    "cardiovascular": {"validity": "说明", "missing_negative_evidence": "说明", "kg_alignment": "说明"},
    "digestive_liver": {"validity": "说明", "missing_negative_evidence": "说明", "kg_alignment": "说明"},
    "endocrine_metabolic": {"validity": "说明", "missing_negative_evidence": "说明", "kg_alignment": "说明"}
  },
  "conflict_resolution": "多专科冲突识别与协调结论",
  "calibrated_confidences": {
    "疾病或风险方向": {"raw_score": 0.0, "calibrated_score": 0.0, "calibration_reason": "理由"}
  },
  "overall_risk": "low/medium/high"
}
"""


SUMMARY_PROMPT = """你是多学科联合会诊 MDT 主审专家 Agent。请基于专科修正意见和 Critique 审计结果，生成最终综合报告。
要求：
1. 不要替代医生诊断。
2. 明确最可能的风险/疾病方向。
3. 列出关键证据、冲突点和建议检查/就医方向。
4. 只返回 JSON，不要 Markdown。

JSON 格式：
{
  "final_diagnosis": "最终风险或疑似疾病方向",
  "final_risk_level": "low/medium/high",
  "final_confidence": 0.0,
  "diagnostic_consensual_reasons": ["理由1"],
  "mdt_consultation_summary": "MDT 综合说明",
  "clinical_intervention_plan": "复查、检查、就医科室和生活方式建议"
}
"""


def run_llm_mdt_agent(
    llm: Any,
    raw_text: str,
    entities: List[Dict[str, Any]],
    retrieved_cases: List[Dict[str, Any]],
    kg_evidence: List[Dict[str, Any]],
    max_cases: int = 3,
    max_kg: int = 5,
) -> Dict[str, Any]:
    """Run the optional LLM-MDT flow and return unified report fields."""

    context = _build_context(
        raw_text=raw_text,
        entities=entities,
        retrieved_cases=retrieved_cases[:max_cases],
        kg_evidence=kg_evidence[:max_kg],
    )

    raw_outputs = _run_specialist_round(llm, context)
    revised_outputs = _run_self_critique_round(llm, context, raw_outputs)
    critique = _call_json(
        llm=llm,
        system_prompt=CRITIQUE_PROMPT,
        user_content=(
            "患者与证据上下文：\n"
            f"{context}\n\n"
            "三位专科 Agent 修正意见：\n"
            f"{json.dumps(revised_outputs, ensure_ascii=False, indent=2)}"
        ),
        mode="MDT-Critique",
    )
    final_report = _call_json(
        llm=llm,
        system_prompt=SUMMARY_PROMPT,
        user_content=(
            "三位专科 Agent 修正意见：\n"
            f"{json.dumps(revised_outputs, ensure_ascii=False, indent=2)}\n\n"
            "Critique 审计结果：\n"
            f"{json.dumps(critique, ensure_ascii=False, indent=2)}"
        ),
        mode="MDT-Summary",
    )

    agent_opinions = [
        _to_agent_opinion(specialty_key, row)
        for specialty_key, row in revised_outputs.items()
    ]

    return {
        "agent_opinions": agent_opinions,
        "critique": {
            "source": "llm_mdt",
            "raw_specialist_opinions": raw_outputs,
            "revised_specialist_opinions": revised_outputs,
            "critique_calibration": critique,
        },
        "mdt_report": {
            "enabled": True,
            "source": "llm_mdt",
            "final_report": final_report,
        },
    }


def _run_specialist_round(llm: Any, context: str) -> Dict[str, Dict[str, Any]]:
    def _call(item: tuple[str, str, str]) -> tuple[str, Dict[str, Any]]:
        key, name, responsibility = item
        prompt = _specialist_prompt(name, responsibility)
        return key, _call_json(llm, prompt, context, mode=f"MDT-{key}-Initial")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SPECIALISTS)) as executor:
        return dict(executor.map(_call, SPECIALISTS))


def _run_self_critique_round(
    llm: Any,
    context: str,
    raw_outputs: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    def _call(item: tuple[str, Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
        key, raw_output = item
        user_content = (
            "患者与证据上下文：\n"
            f"{context}\n\n"
            "你的初步评估：\n"
            f"{json.dumps(raw_output, ensure_ascii=False, indent=2)}"
        )
        return key, _call_json(llm, SELF_CRITIQUE_PROMPT, user_content, mode=f"MDT-{key}-SelfCritique")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(raw_outputs) or 1) as executor:
        return dict(executor.map(_call, raw_outputs.items()))


def _specialist_prompt(name: str, responsibility: str) -> str:
    return f"""你是一位资深的{name}专家 Agent。你的职责是：{responsibility}
请基于患者文本、结构化指标、相似病例和知识图谱证据给出初步评估。
要求：
1. 只使用输入中存在的证据。
2. 不要把相似病例当成确诊。
3. 只返回 JSON，不要 Markdown。

JSON 格式：
{{
  "specialty": "{name}",
  "suspected_disease": "疑似疾病或风险方向；没有则填无明确专科风险",
  "risk_level": "low/medium/high",
  "confidence_score": 0.0,
  "supporting_evidence": ["支持证据1"],
  "exclusion_evidence": ["排除证据1"],
  "suggestion": "建议复查或就医方向"
}}
"""


def _build_context(
    raw_text: str,
    entities: List[Dict[str, Any]],
    retrieved_cases: List[Dict[str, Any]],
    kg_evidence: List[Dict[str, Any]],
) -> str:
    compact_cases = [
        {
            "case_id": row.get("case_id", ""),
            "source": row.get("source", ""),
            "diagnosis": row.get("diagnosis") or row.get("title", ""),
            "similarity": row.get("similarity", row.get("score", 0.0)),
            "raw_text": _short_text(row.get("raw_text") or row.get("text") or "", 500),
        }
        for row in retrieved_cases
    ]
    compact_kg = [
        {
            "head": row.get("head", ""),
            "relation": row.get("relation", ""),
            "tail": row.get("tail", ""),
            "relation_category": row.get("relation_category", ""),
            "score": row.get("score", 0.0),
        }
        for row in kg_evidence
    ]
    return (
        "患者文本：\n"
        f"{_short_text(raw_text, 2000)}\n\n"
        "结构化医学指标 entities：\n"
        f"{json.dumps(entities, ensure_ascii=False, indent=2)[:3000]}\n\n"
        "相似病例证据 retrieved_cases：\n"
        f"{json.dumps(compact_cases, ensure_ascii=False, indent=2)[:3000]}\n\n"
        "知识图谱证据 kg_evidence：\n"
        f"{json.dumps(compact_kg, ensure_ascii=False, indent=2)[:3000]}"
    )


def _call_json(llm: Any, system_prompt: str, user_content: str, mode: str) -> Dict[str, Any]:
    response = llm.generate(
        user_content,
        system_prompt=system_prompt,
        mode=mode,
    )
    parsed = _parse_json_object(response)
    if parsed is not None:
        return parsed
    return {
        "parse_error": True,
        "raw_response": str(response or ""),
    }


def _parse_json_object(text: Any) -> Dict[str, Any] | None:
    value = str(text or "").strip()
    if not value:
        return None
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    candidate = _extract_json_object(value)
    if not candidate:
        return None
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _to_agent_opinion(specialty_key: str, row: Dict[str, Any]) -> Dict[str, Any]:
    confidence = _float_or_default(
        row.get("revised_confidence", row.get("confidence_score", 0.0)),
        0.0,
    )
    risk_level = str(row.get("revised_risk_level") or row.get("risk_level") or _risk_from_confidence(confidence))
    diagnosis = str(row.get("revised_suspected_disease") or row.get("suspected_disease") or "").strip()
    evidence = row.get("revised_supporting_evidence") or row.get("supporting_evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    return {
        "source": "llm_mdt",
        "specialty": specialty_key,
        "agent_name": str(row.get("specialty") or specialty_key),
        "risk_level": risk_level,
        "confidence": confidence,
        "diagnosis": [diagnosis] if diagnosis else [],
        "evidence": evidence if isinstance(evidence, list) else [],
        "suggestion": str(row.get("suggestion") or row.get("self_critique_log") or ""),
        "raw": row,
    }


def _risk_from_confidence(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _short_text(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."
