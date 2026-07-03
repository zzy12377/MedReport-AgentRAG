# -*- coding: utf-8 -*-
"""Report persistence helpers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from backend.app.config.settings import settings


class ReportService:
    def __init__(self, report_dir: str | None = None) -> None:
        self.report_dir = os.path.normpath(report_dir or settings.report_dir)
        os.makedirs(self.report_dir, exist_ok=True)

    def report_path(self, task_id: str) -> str:
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(task_id))
        return os.path.join(self.report_dir, f"{safe_id}.json")

    def markdown_path(self, task_id: str) -> str:
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(task_id))
        return os.path.join(self.report_dir, f"{safe_id}.md")

    def save_report(self, task_id: str, report: Dict[str, Any]) -> str:
        path = self.report_path(task_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        markdown = str(report.get("summary_markdown") or "").strip()
        if markdown:
            with open(self.markdown_path(task_id), "w", encoding="utf-8") as f:
                f.write(markdown)
        return path

    def load_report(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = self.report_path(task_id)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_reports(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not os.path.isdir(self.report_dir):
            return items
        for name in sorted(os.listdir(self.report_dir), reverse=True):
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.report_dir, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                items.append({"task_id": obj.get("task_id") or name[:-5], "report": obj, "path": path})
            except Exception:
                continue
        return items
