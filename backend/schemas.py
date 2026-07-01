# -*- coding: utf-8 -*-
"""Backend data schemas kept dependency-light for phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class AnalyzeRequest:
    task_id: str
    mode: str = "B1"
    text: str = ""


@dataclass
class AnalyzeResult:
    task_id: str
    status: str
    result: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

