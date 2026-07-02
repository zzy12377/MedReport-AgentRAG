# -*- coding: utf-8 -*-
"""Task and report query routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.cache.memory_store import MemoryStore
from backend.app.services.report_service import ReportService

router = APIRouter()
store = MemoryStore.get_instance()
reports = ReportService()


@router.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    task = store.get_task(task_id)
    if task:
        return task
    report = reports.load_report(task_id)
    if report:
        return {
            "task_id": task_id,
            "status": "done",
            "progress": 100,
            "message": "Loaded from report storage",
            "result": report,
            "error": None,
        }
    raise HTTPException(status_code=404, detail="Task not found")


@router.get("/diagnosis/{task_id}/report")
def get_report(task_id: str) -> dict:
    task = store.get_task(task_id)
    if task and task.get("status") != "done":
        return {
            "task_id": task_id,
            "status": task.get("status"),
            "message": "Task is not finished",
        }
    if task and task.get("result"):
        return task["result"]
    report = reports.load_report(task_id)
    if report:
        return report
    raise HTTPException(status_code=404, detail="Report not found")
