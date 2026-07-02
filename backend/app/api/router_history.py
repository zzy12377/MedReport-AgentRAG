# -*- coding: utf-8 -*-
"""History routes."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.cache.memory_store import MemoryStore
from backend.app.services.report_service import ReportService

router = APIRouter()
store = MemoryStore.get_instance()
reports = ReportService()


@router.get("/history")
def list_history() -> dict:
    items = store.list_history()
    persisted = reports.list_reports()
    known = {item.get("task_id") for item in items}
    for item in persisted:
        if item.get("task_id") not in known:
            items.append(item)
    return {"items": items}


@router.get("/history/{task_id}")
def get_history(task_id: str) -> dict:
    item = store.get_history(task_id)
    if item:
        return item
    report = reports.load_report(task_id)
    if report:
        return {"task_id": task_id, "report": report}
    return {"task_id": task_id, "message": "History item not found"}
