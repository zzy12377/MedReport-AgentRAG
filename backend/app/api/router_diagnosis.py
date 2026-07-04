# -*- coding: utf-8 -*-
"""Diagnosis API routes."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, Body, File, Form, HTTPException, UploadFile

from backend.app.schemas.patient_case import DiagnosisRequest
from backend.app.config.settings import settings
from backend.app.services.pipeline import DiagnosisPipeline
from backend.app.services.task_service import TaskService
from backend.app.services.ocr_json_service import normalize_ocr_json, split_ocr_request_payload

router = APIRouter()
task_service = TaskService()


@router.post("/diagnosis/text")
async def diagnose_text(req: DiagnosisRequest, background_tasks: BackgroundTasks) -> dict:
    task_id = task_service.create_diagnosis_task(
        background_tasks=background_tasks,
        text=req.text,
        top_k=req.top_k,
        use_multi_agent=req.use_multi_agent,
        use_kg=req.use_kg,
        vector_sources=req.vector_sources,
    )
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Diagnosis task submitted",
    }


@router.post("/diagnosis/text/sync")
async def diagnose_text_sync(req: DiagnosisRequest) -> dict:
    report = await DiagnosisPipeline().run(
        raw_text=req.text,
        top_k=req.top_k,
        use_multi_agent=req.use_multi_agent,
        use_kg=req.use_kg,
        vector_sources=req.vector_sources,
    )
    return {
        "status": "done",
        "report": report,
    }


@router.post("/diagnosis/ocr-json")
async def diagnose_ocr_json(background_tasks: BackgroundTasks, payload: Any = Body(...)) -> dict:
    ocr_json, options = _split_payload(payload)
    normalized = _normalize_or_400(ocr_json)
    task_id = task_service.create_diagnosis_task(
        background_tasks=background_tasks,
        text=normalized["text"],
        top_k=_int_option(options.get("top_k"), default=3),
        use_multi_agent=_bool_option(options.get("use_multi_agent"), default=True),
        use_kg=_bool_option(options.get("use_kg"), default=True),
        vector_sources=_vector_sources_option(options.get("vector_sources")),
    )
    return {
        "task_id": task_id,
        "status": "pending",
        "input_type": "ocr_json",
        "normalized_input": _normalized_preview(normalized, options),
        "message": "OCR JSON diagnosis task submitted",
    }


@router.post("/diagnosis/ocr-json/sync")
async def diagnose_ocr_json_sync(payload: Any = Body(...)) -> dict:
    ocr_json, options = _split_payload(payload)
    normalized = _normalize_or_400(ocr_json)
    report = await DiagnosisPipeline().run(
        raw_text=normalized["text"],
        top_k=_int_option(options.get("top_k"), default=3),
        use_multi_agent=_bool_option(options.get("use_multi_agent"), default=True),
        use_kg=_bool_option(options.get("use_kg"), default=True),
        vector_sources=_vector_sources_option(options.get("vector_sources")),
    )
    case_id = options.get("case_id")
    if case_id:
        report["case_id"] = str(case_id)
    return {
        "status": "done",
        "input_type": "ocr_json",
        "normalized_input": _normalized_preview(normalized, options),
        "report": report,
    }


@router.post("/diagnosis/ocr-json-file/sync")
async def diagnose_ocr_json_file_sync(
    file: UploadFile = File(...),
    top_k: int = Form(3),
    use_multi_agent: bool = Form(True),
    use_kg: bool = Form(True),
    vector_sources: Optional[str] = Form(None),
    case_id: Optional[str] = Form(None),
) -> dict:
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json OCR files are supported by this endpoint.")
    raw = await file.read()
    try:
        ocr_json = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid OCR JSON file: {exc}") from exc

    os.makedirs(settings.upload_dir, exist_ok=True)
    save_name = f"{uuid.uuid4()}_{os.path.basename(file.filename or 'ocr.json')}"
    save_path = os.path.join(settings.upload_dir, save_name)
    with open(save_path, "wb") as f:
        f.write(raw)

    normalized = _normalize_or_400(ocr_json)
    options = {
        "case_id": case_id,
        "top_k": top_k,
        "use_multi_agent": use_multi_agent,
        "use_kg": use_kg,
        "vector_sources": vector_sources,
    }
    report = await DiagnosisPipeline().run(
        raw_text=normalized["text"],
        top_k=top_k,
        use_multi_agent=use_multi_agent,
        use_kg=use_kg,
        vector_sources=_vector_sources_option(vector_sources),
    )
    if case_id:
        report["case_id"] = str(case_id)
    return {
        "status": "done",
        "input_type": "ocr_json_file",
        "file_path": save_path,
        "normalized_input": _normalized_preview(normalized, options),
        "report": report,
    }


def _split_payload(payload: Any) -> tuple[Any, Dict[str, Any]]:
    if isinstance(payload, dict):
        return split_ocr_request_payload(payload)
    return payload, {}


def _normalize_or_400(ocr_json: Any) -> Dict[str, Any]:
    normalized = normalize_ocr_json(ocr_json)
    if not normalized.get("text"):
        raise HTTPException(
            status_code=400,
            detail="OCR JSON 中没有可用于诊断的文本。请确认包含 text、ocr_text、pages、lines、blocks 或 results 字段。",
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
