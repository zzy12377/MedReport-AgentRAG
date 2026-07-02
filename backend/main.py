# -*- coding: utf-8 -*-
"""Legacy FastAPI entrypoint.

The course documents now use ``backend.app.main:app``. This file remains as a
compatibility proxy so existing imports do not break.
"""

from __future__ import annotations


try:
    from backend.app.main import app

except Exception as exc:  # pragma: no cover - optional dependency path
    app = None
    FASTAPI_IMPORT_ERROR = str(exc)
