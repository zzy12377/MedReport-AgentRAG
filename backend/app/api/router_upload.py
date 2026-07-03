# -*- coding: utf-8 -*-
"""Upload routes."""

from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, BackgroundTasks, File, UploadFile

from backend.app.config.settings import settings
from backend.app.services.ocr_service import OCRService
from backend.app.services.task_service import TaskService

router = APIRouter()
task_service = TaskService()


@router.post("/reports/upload")
async def upload_report(background_tasks: BackgroundTasks, file: UploadFile = File(...)) -> dict:
    os.makedirs(settings.upload_dir, exist_ok=True)
    ext = os.path.splitext(file.filename or "")[-1]
    file_id = str(uuid.uuid4())
    save_name = f"{file_id}{ext}"
    save_path = os.path.join(settings.upload_dir, save_name)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    ocr_service = OCRService()
    ocr_result = ocr_service.extract(save_path)
    ocr_text = str(ocr_result.get("text") or "").strip()
    ocr_text_path = ocr_service.save_ocr_text(file_id, ocr_text)
    if not ocr_text:
        return {
            "task_id": None,
            "status": "ocr_failed",
            "file_path": save_path,
            "ocr_text_path": ocr_text_path,
            "ocr_text_preview": "",
            "ocr_result": ocr_result,
            "message": ocr_result.get("warning") or "No text extracted from uploaded file.",
        }
    task_id = task_service.create_diagnosis_task(
        background_tasks=background_tasks,
        text=ocr_text,
        top_k=settings.default_top_k,
        use_multi_agent=True,
        use_kg=True,
        vector_sources=["all"],
    )
    return {
        "task_id": task_id,
        "status": "pending",
        "file_path": save_path,
        "ocr_text_path": ocr_text_path,
        "ocr_text_preview": ocr_text[:500],
        "ocr_result": {k: v for k, v in ocr_result.items() if k != "pages"},
        "message": "File uploaded and diagnosis task submitted",
    }
