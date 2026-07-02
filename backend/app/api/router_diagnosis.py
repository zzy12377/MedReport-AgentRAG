# -*- coding: utf-8 -*-
"""Diagnosis API routes."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from backend.app.schemas.patient_case import DiagnosisRequest
from backend.app.services.pipeline import DiagnosisPipeline
from backend.app.services.task_service import TaskService

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
