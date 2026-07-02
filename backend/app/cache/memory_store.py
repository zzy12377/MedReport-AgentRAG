# -*- coding: utf-8 -*-
"""Small in-memory store used before Redis is introduced."""

from __future__ import annotations

from threading import RLock
from typing import Any, Dict, List, Optional


class MemoryStore:
    _instance: "MemoryStore | None" = None

    def __init__(self) -> None:
        self._lock = RLock()
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.history: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def get_instance(cls) -> "MemoryStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_task(self, task_id: str, data: Dict[str, Any]) -> None:
        with self._lock:
            self.tasks[task_id] = dict(data)

    def update_task(self, task_id: str, **updates: Any) -> Dict[str, Any]:
        with self._lock:
            row = self.tasks.setdefault(task_id, {"task_id": task_id})
            row.update(updates)
            return dict(row)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            task = self.tasks.get(task_id)
            return dict(task) if task is not None else None

    def add_history(self, task_id: str, report: Dict[str, Any]) -> None:
        with self._lock:
            self.history[task_id] = {"task_id": task_id, "report": report}

    def get_history(self, task_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self.history.get(task_id)
            return dict(item) if item is not None else None

    def list_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self.history.values())
