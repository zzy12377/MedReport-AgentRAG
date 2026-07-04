# -*- coding: utf-8 -*-
"""Asynchronous task service."""

from __future__ import annotations

import traceback
import uuid
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks

from backend.app.cache.memory_store import MemoryStore
from backend.app.services.pipeline import DiagnosisPipeline
from backend.app.services.report_service import ReportService


class TaskService:
    def __init__(self) -> None:
        self.store = MemoryStore.get_instance()
        self.report_service = ReportService()

    def create_diagnosis_task(
        self,
        background_tasks: BackgroundTasks,
        text: str,
        top_k: int = 3,
        use_multi_agent: bool = True,
        use_kg: bool = True,
        vector_sources: Optional[list[str]] = None,
        case_id: Optional[str] = None,
        input_type: str = "text",
        normalized_input: Optional[Dict[str, Any]] = None,
    ) -> str:
        task_id = str(uuid.uuid4())
        self.store.set_task(
            task_id,
            {
                "task_id": task_id,
                "status": "pending",
                "progress": 0,
                "message": "Task created",
                "case_id": case_id,
                "input_type": input_type,
                "normalized_input": normalized_input or {},
                "result": None,
                "error": None,
            },
        )
        background_tasks.add_task(
            self._run_task,
            task_id,
            text,
            top_k,
            use_multi_agent,
            use_kg,
            vector_sources,
            case_id,
            input_type,
            normalized_input or {},
        )
        return task_id

    async def _run_task(
        self,
        task_id: str,
        text: str,
        top_k: int,
        use_multi_agent: bool,
        use_kg: bool,
        vector_sources: Optional[list[str]],
        case_id: Optional[str],
        input_type: str,
        normalized_input: Dict[str, Any],
    ) -> None:
        try:
            self.store.update_task(task_id, status="extracting", progress=10, message="Extracting medical entities")
            pipeline = DiagnosisPipeline()
            self.store.update_task(task_id, status="diagnosing", progress=40, message="Retrieving evidence")
            report = await pipeline.run(
                raw_text=text,
                top_k=top_k,
                use_multi_agent=use_multi_agent,
                use_kg=use_kg,
                vector_sources=vector_sources,
            )
            if case_id:
                report["case_id"] = str(case_id)
            report["input_type"] = input_type
            report["normalized_input"] = normalized_input
            report["task_id"] = task_id
            path = self.report_service.save_report(task_id, report)
            self.store.add_history(task_id, report)
            self.store.update_task(
                task_id,
                status="done",
                progress=100,
                message="Diagnosis completed",
                result=report,
                report_path=path,
                error=None,
            )
        except Exception as exc:
            self.store.update_task(
                task_id,
                status="failed",
                progress=100,
                message="Task failed",
                result=None,
                error=str(exc) + "\n" + traceback.format_exc(),
            )
