# -*- coding: utf-8 -*-
"""Task status schema."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    message: str = ""
    result: Optional[Any] = None
    error: Optional[str] = None
