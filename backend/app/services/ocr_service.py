# -*- coding: utf-8 -*-
"""OCR service wrapper.

Text files are read directly. Images/PDFs go through the optional OCR engine; if
PaddleOCR is unavailable the service returns an explicit placeholder message.
"""

from __future__ import annotations

import os

from backend.app.core.ocr.paddle_ocr_engine import PaddleOCREngine


class OCRService:
    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".txt", ".md", ".csv"}:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()

        result = PaddleOCREngine().extract_text(file_path)
        text = str(result.get("text") or "").strip()
        if text:
            return text
        warning = result.get("warning") or "OCR engine unavailable."
        return f"[OCR unavailable] {warning}"
