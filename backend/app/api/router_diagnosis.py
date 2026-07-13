# -*- coding: utf-8 -*-
"""Diagnosis and report-generation API routes."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import PlainTextResponse

from backend.app.services.ocr_json_service import normalize_ocr_json, split_ocr_request_payload
from backend.app.services.pipeline import DiagnosisPipeline
from backend.app.services.report_service import ReportService

router = APIRouter()
report_service = ReportService()


@router.post("/diagnosis/ocr-json/sync")
async def diagnose_ocr_json_sync(payload: Any = Body(...)) -> dict:
    full_response = await _diagnose_ocr_json_sync_impl(payload, input_type="ocr_json")
    if _wants_markdown_response(payload):
        return _markdown_response_payload(full_response)
    return full_response


@router.post("/reports/from-ocr-json")
async def create_report_from_ocr_json(payload: Any = Body(...)) -> dict:
    """Frontend-friendly alias: OCR JSON in, saved diagnosis report out."""

    full_response = await _diagnose_ocr_json_sync_impl(payload, input_type="ocr_json")
    return _markdown_response_payload(full_response)


@router.post("/reports/from-ocr-json/simple")
async def create_simple_report_from_ocr_json(payload: Any = Body(...)) -> dict:
    """OCR JSON in, page-ready report text out.

    This endpoint keeps the full report saved on disk but returns only the
    fields a frontend normally needs for direct display.
    """

    full_response = await _diagnose_ocr_json_sync_impl(payload, input_type="ocr_json")
    return _markdown_response_payload(full_response)


@router.post("/reports/from-ocr-json/markdown", response_class=PlainTextResponse)
async def create_markdown_report_from_ocr_json(payload: Any = Body(...)) -> PlainTextResponse:
    """OCR JSON in, raw Markdown report body out."""

    full_response = await _diagnose_ocr_json_sync_impl(payload, input_type="ocr_json")
    return PlainTextResponse(_extract_display_report_text(full_response))


async def _diagnose_ocr_json_sync_impl(payload: Any, input_type: str) -> dict:
    ocr_json, options = _split_payload(payload)
    normalized = _normalize_or_400(ocr_json)
    report = await DiagnosisPipeline().run(
        raw_text=normalized["text"],
        top_k=_int_option(options.get("top_k"), default=5),
        use_multi_agent=_bool_option(options.get("use_multi_agent"), default=True),
        use_kg=_bool_option(options.get("use_kg"), default=True),
        vector_sources=_vector_sources_option(options.get("vector_sources")),
        baseline_modes=["B2"],
    )
    case_id = options.get("case_id")
    if case_id:
        report["case_id"] = str(case_id)
    normalized_preview = _normalized_preview(normalized, options)
    saved = _save_sync_report(
        report,
        input_type=input_type,
        normalized_input=normalized_preview,
    )
    return {
        "status": "done",
        "task_id": report.get("task_id"),
        "report_id": report.get("task_id"),
        "report_path": saved["report_path"],
        "input_type": input_type,
        "normalized_input": normalized_preview,
        "report": report,
    }


def _save_sync_report(
    report: Dict[str, Any],
    input_type: str,
    normalized_input: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_id = str(report.get("task_id") or uuid.uuid4())
    report["task_id"] = task_id
    report["input_type"] = input_type
    if normalized_input is not None:
        report["normalized_input"] = normalized_input
    report_path = report_service.save_report(task_id, report)
    return {"report_path": report_path}


def _extract_display_report_text(response: Dict[str, Any]) -> str:
    report = response.get("report") or {}
    text = report.get("summary_markdown")
    if text:
        return str(text)

    conclusion = report.get("detection_conclusion") or report.get("primary_diagnosis") or "未生成明确结论"
    safety_note = report.get("safety_note") or "本结果仅用于课程演示和辅助参考，不能替代医生诊断。"
    return f"## 检测报告\n\n### 一、检测结论\n{conclusion}\n\n### 二、安全提示\n{safety_note}"


def _markdown_response_payload(full_response: Dict[str, Any]) -> Dict[str, Any]:
    report = full_response.get("report") or {}
    return {
        "status": full_response.get("status", "done"),
        "task_id": full_response.get("task_id"),
        "report_id": full_response.get("report_id"),
        "report_path": _public_path(full_response.get("report_path")),
        "format": "markdown",
        "report_text": _extract_display_report_text(full_response),
        "visualization": _visualization_payload(report),
    }


def _visualization_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "summary": _visual_summary(report),
        "indicators": _visual_indicators(report),
        "agents": _visual_agents(report),
        "similar_cases": _visual_similar_cases(report),
        "knowledge_graph": _visual_knowledge_graph(report),
        "knowledge": _visual_knowledge(report),
        "recommendation": _visual_recommendation(report),
        "warning": report.get("safety_note") or "本结果仅用于课程演示和辅助参考，不能替代医生诊断。",
    }


def _visual_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    conclusion = report.get("detection_conclusion") if isinstance(report.get("detection_conclusion"), dict) else {}
    confidence = _safe_float(conclusion.get("confidence"), 0.0)
    return {
        "title": "医疗检测报告",
        "diagnosis": str(conclusion.get("primary_diagnosis") or "待进一步临床确认"),
        "risk_level": _risk_label(conclusion.get("overall_risk") or report.get("overall_risk")),
        "confidence": confidence,
        "reason": str(conclusion.get("basis") or ""),
        "health_score": _health_score(report, confidence),
    }


def _visual_indicators(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entity in report.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        low = entity.get("ref_low")
        high = entity.get("ref_high")
        rows.append(
            {
                "name": _indicator_display_name(entity.get("name")),
                "abbreviation": _indicator_abbreviation(entity.get("name")),
                "value": entity.get("value"),
                "unit": entity.get("unit") or "",
                "reference": _reference_range(low, high),
                "status": _indicator_status(entity),
            }
        )
    return rows


def _visual_agents(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in report.get("agent_opinions") or []:
        if not isinstance(item, dict):
            continue
        department = _department_name(item.get("specialty") or item.get("agent_name"))
        risk_level = _risk_label(item.get("risk_level"))
        rows.append(
            {
                "department": department,
                "risk_level": risk_level,
                "confidence": _safe_float(item.get("confidence"), 0.0),
                "summary": _agent_summary(department, risk_level),
                "recommendation": _agent_recommendation(department, risk_level),
            }
        )
    return rows


def _visual_similar_cases(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in (report.get("retrieved_cases") or [])[:5]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "case_id": item.get("case_id") or item.get("id") or "",
                "disease": item.get("diagnosis") or item.get("title") or "",
                "similarity": round(_safe_float(item.get("similarity", item.get("score")), 0.0), 4),
            }
        )
    return rows


def _visual_knowledge_graph(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in (report.get("kg_evidence") or [])[:8]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "disease": item.get("head") or "",
                "relation": item.get("relation_category") or item.get("relation") or "",
                "entity": item.get("relation") or "",
                "value": item.get("tail") or "",
            }
        )
    return rows


def _visual_knowledge(report: Dict[str, Any]) -> List[Dict[str, str]]:
    diagnosis = str((report.get("detection_conclusion") or {}).get("primary_diagnosis") or "")
    text = diagnosis.lower()
    if "血压" in diagnosis or "高血压" in diagnosis or "hypertension" in text:
        return [
            {"source": "中国高血压防治指南", "content": "持续血压≥140/90 mmHg 建议进一步评估及干预。"},
            {"source": "中国居民膳食指南", "content": "每日食盐摄入量建议低于5g。"},
        ]
    if "血糖" in diagnosis or "糖" in diagnosis or "glucose" in text or "diabetes" in text:
        return [
            {"source": "中国2型糖尿病防治指南", "content": "血糖异常建议结合空腹血糖、餐后血糖和糖化血红蛋白综合评估。"},
            {"source": "中国居民膳食指南", "content": "建议控制精制糖摄入并保持规律运动。"},
        ]
    if "肝" in diagnosis or "alt" in text or "ast" in text or "liver" in text:
        return [
            {"source": "临床肝功能检查解读", "content": "转氨酶升高时建议结合用药史、饮酒史、病毒性肝炎筛查和影像学检查综合判断。"},
        ]
    return [
        {"source": "医疗辅助诊断提示", "content": "报告结果需要结合病史、体征、复查结果和临床医生判断。"},
    ]


def _visual_recommendation(report: Dict[str, Any]) -> Dict[str, List[str]]:
    indicators = report.get("entities") or []
    names = {str(item.get("name") or "").lower() for item in indicators if isinstance(item, dict)}
    diagnosis = str((report.get("detection_conclusion") or {}).get("primary_diagnosis") or "").lower()
    has_bp = any("血压" in name or "sbp" in name or "dbp" in name or "鏀剁缉" in name or "鑸掑紶" in name for name in names) or "hypertension" in diagnosis
    has_glucose = any(name in {"glu", "hba1c"} or "血糖" in name for name in names) or "diabetes" in diagnosis
    has_liver = any(name in {"alt", "ast", "ggt", "tbil", "alp"} for name in names) or "liver" in diagnosis
    diet = ["保持均衡饮食"]
    exercise = ["保持规律运动"]
    follow_up = ["如有不适或指标持续异常，建议咨询医生"]
    if has_bp:
        diet = ["控制食盐摄入", "减少高脂食物"]
        exercise = ["每周中等强度运动150分钟"]
        follow_up = ["建议1个月后复查血压"]
    elif has_glucose:
        diet = ["控制精制糖和高热量食物摄入", "规律进餐"]
        exercise = ["餐后适量活动", "每周规律有氧运动"]
        follow_up = ["建议复查空腹血糖和糖化血红蛋白"]
    elif has_liver:
        diet = ["避免饮酒", "减少油腻饮食"]
        exercise = ["保持适量运动，避免过度劳累"]
        follow_up = ["建议复查肝功能并结合医生评估"]
    return {"diet": diet, "exercise": exercise, "follow_up": follow_up}


def _public_path(value: Any) -> str:
    return str(value or "").replace("\\", "/").lstrip("./")


def _risk_label(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {"high": "高", "medium": "中", "low": "低", "unknown": "未知"}.get(text, str(value or "未知"))


def _indicator_display_name(value: Any) -> str:
    text = str(value or "").strip()
    mapping = {
        "SBP": "收缩压",
        "DBP": "舒张压",
        "收缩压": "收缩压",
        "舒张压": "舒张压",
        "心率": "心率",
        "鏀剁缉鍘?": "收缩压",
        "鑸掑紶鍘?": "舒张压",
        "蹇冪巼": "心率",
    }
    return mapping.get(text, text)


def _indicator_abbreviation(value: Any) -> str:
    name = _indicator_display_name(value)
    mapping = {"收缩压": "SBP", "舒张压": "DBP", "心率": "HR"}
    return mapping.get(name, str(value or ""))


def _reference_range(low: Any, high: Any) -> str:
    if low in (None, "") or high in (None, ""):
        return ""
    return f"{_compact_number(low)}~{_compact_number(high)}"


def _indicator_status(entity: Dict[str, Any]) -> str:
    value = _safe_float(entity.get("value"), 0.0)
    low = entity.get("ref_low")
    high = entity.get("ref_high")
    if high not in (None, "") and value > _safe_float(high, value):
        return "high"
    if low not in (None, "") and value < _safe_float(low, value):
        return "low"
    return "normal"


def _department_name(value: Any) -> str:
    text = str(value or "").strip()
    key = text.lower()
    if "cardio" in key or "心血管" in text or "蹇冭" in text:
        return "心血管"
    if "liver" in key or "digestive" in key or "肝" in text or "鑲" in text:
        return "肝脏"
    if "endocrine" in key or "metabolic" in key or "内分泌" in text:
        return "内分泌"
    return text or "综合"


def _agent_summary(department: str, risk_level: str) -> str:
    if risk_level == "高":
        return f"{department}相关风险较高，建议尽快结合临床复查。"
    if risk_level == "中":
        return f"{department}相关风险需要关注，建议结合症状和复查结果判断。"
    return f"{department}未见明显高风险信号。"


def _agent_recommendation(department: str, risk_level: str) -> List[str]:
    if department == "心血管":
        return ["减少钠盐摄入", "规律运动", "建议复查血压"]
    if department == "肝脏":
        return ["避免饮酒", "保持健康饮食", "必要时复查肝功能"]
    if department == "内分泌":
        return ["保持正常体重", "控制精制糖摄入", "必要时复查血糖"]
    return ["结合医生建议进一步评估"] if risk_level in {"中", "高"} else ["保持健康生活方式"]


def _health_score(report: Dict[str, Any], confidence: float) -> int:
    abnormal_count = len([item for item in report.get("entities") or [] if isinstance(item, dict) and item.get("is_abnormal")])
    risk = str(report.get("overall_risk") or "").lower()
    penalty = abnormal_count * 6
    if risk == "high":
        penalty += 10
    elif risk == "medium":
        penalty += 5
    score = round(100 - penalty - max(0.0, confidence - 0.8) * 10)
    return max(0, min(100, int(score)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _compact_number(value: Any) -> str:
    number = _safe_float(value, 0.0)
    return str(int(number)) if number.is_integer() else str(number)


def _wants_markdown_response(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    value = payload.get("response_format") or payload.get("report_format") or payload.get("output_format")
    return str(value or "").strip().lower() in {"markdown", "md", "text/markdown"}


def _split_payload(payload: Any) -> tuple[Any, Dict[str, Any]]:
    if isinstance(payload, dict):
        return split_ocr_request_payload(payload)
    return payload, {}


def _normalize_or_400(ocr_json: Any) -> Dict[str, Any]:
    normalized = normalize_ocr_json(ocr_json)
    if not normalized.get("text"):
        raise HTTPException(
            status_code=400,
            detail=(
                "OCR JSON 中没有可用于诊断的文本。请确认包含 text、ocr_text、"
                "plain_text、pages、lines、blocks 或 results 字段。"
            ),
        )
    return normalized


def _normalized_preview(normalized: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    text = str(normalized.get("text") or "")
    return {
        "case_id": options.get("case_id"),
        "source_format": normalized.get("source_format"),
        "line_count": normalized.get("line_count", 0),
        "text": text,
        "text_preview": text[:500],
        "interpretive_notes": normalized.get("interpretive_notes", []),
    }


def _int_option(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _bool_option(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _vector_sources_option(value: Any) -> Optional[list[str]]:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _baseline_modes_option(value: Any) -> Optional[list[str]]:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).replace("+", ",").replace("/", ",").replace("|", ",")
    return [item.strip() for item in text.split(",") if item.strip()]


def _first_option(options: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in options:
            return options.get(key)
    return None
