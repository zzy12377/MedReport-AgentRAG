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
    save_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(settings.upload_dir, save_name)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    ocr_text = OCRService().extract_text(save_path)
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
        "ocr_text_preview": ocr_text[:500],
        "message": "File uploaded and diagnosis task submitted",
    }
