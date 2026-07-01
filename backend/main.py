# -*- coding: utf-8 -*-
"""FastAPI entrypoint.

FastAPI is optional in phase 1. Importing this module should not fail if the
web stack is not installed.
"""

from __future__ import annotations


try:
    from fastapi import FastAPI

    app = FastAPI(title="MedReport AgentRAG", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "phase": "baseline"}

except Exception as exc:  # pragma: no cover - optional dependency path
    app = None
    FASTAPI_IMPORT_ERROR = str(exc)

