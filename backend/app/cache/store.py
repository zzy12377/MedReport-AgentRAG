# -*- coding: utf-8 -*-
"""Cache store factory with optional Redis support.

Redis is deliberately optional: local demos keep working with the in-memory
store when redis-py is not installed or REDIS_URL is not configured.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from backend.app.cache.memory_store import MemoryStore
from backend.app.config.settings import settings


class RedisStore:
    def __init__(self, url: str, ttl_seconds: int = 86400) -> None:
        import redis

        self.client = redis.Redis.from_url(url, decode_responses=True)
        self.ttl_seconds = int(ttl_seconds)
        self.client.ping()

    @staticmethod
    def _task_key(task_id: str) -> str:
        return f"medrag:task:{task_id}"

    @staticmethod
    def _history_key(task_id: str) -> str:
        return f"medrag:history:{task_id}"

    def set_task(self, task_id: str, data: Dict[str, Any]) -> None:
        self.client.setex(self._task_key(task_id), self.ttl_seconds, json.dumps(data, ensure_ascii=False))

    def update_task(self, task_id: str, **updates: Any) -> Dict[str, Any]:
        row = self.get_task(task_id) or {"task_id": task_id}
        row.update(updates)
        self.set_task(task_id, row)
        return row

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.get(self._task_key(task_id))
        return json.loads(raw) if raw else None

    def add_history(self, task_id: str, report: Dict[str, Any]) -> None:
        item = {"task_id": task_id, "report": report}
        self.client.setex(self._history_key(task_id), self.ttl_seconds, json.dumps(item, ensure_ascii=False))

    def get_history(self, task_id: str) -> Optional[Dict[str, Any]]:
        raw = self.client.get(self._history_key(task_id))
        return json.loads(raw) if raw else None

    def list_history(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for key in self.client.scan_iter("medrag:history:*"):
            raw = self.client.get(key)
            if not raw:
                continue
            try:
                items.append(json.loads(raw))
            except Exception:
                continue
        return items


_store: Any = None


def get_store() -> Any:
    global _store
    if _store is not None:
        return _store
    if settings.redis_url:
        try:
            _store = RedisStore(settings.redis_url, ttl_seconds=settings.task_ttl_seconds)
            print(f"[INFO] Using Redis task store: {settings.redis_url}")
            return _store
        except Exception as exc:
            print(f"[WARN] Redis unavailable, falling back to in-memory task store: {exc}")
    _store = MemoryStore.get_instance()
    return _store
