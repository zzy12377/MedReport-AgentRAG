# -*- coding: utf-8 -*-
"""In-memory task state for phase 1."""

from __future__ import annotations

from typing import Dict


TASKS: Dict[str, dict] = {}


def create_task(task_id: str) -> dict:
    TASKS[task_id] = {"task_id": task_id, "status": "pending", "progress": 0}
    return TASKS[task_id]


def update_task(task_id: str, **updates) -> dict:
    TASKS.setdefault(task_id, {"task_id": task_id})
    TASKS[task_id].update(updates)
    return TASKS[task_id]

