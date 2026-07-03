# -*- coding: utf-8 -*-
"""Optional PaddleOCR adapter."""

from __future__ import annotations

from engines.ocr.ocr_engine import OCREngine
from backend.app.config.settings import settings


class PaddleOCREngine(OCREngine):
    def __init__(self) -> None:
        super().__init__(use_paddle=settings.enable_paddleocr)
