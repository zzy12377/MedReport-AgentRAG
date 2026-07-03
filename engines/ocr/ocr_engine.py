# -*- coding: utf-8 -*-
"""Optional OCR wrapper.

Phase 1 does not require PaddleOCR. If PaddleOCR is unavailable, the module
still imports and returns a clear message.
"""

from __future__ import annotations

from typing import Dict, List


class OCREngine:
    def __init__(self, use_paddle: bool = True):
        self.use_paddle = use_paddle
        self._ocr = None
        if use_paddle:
            try:
                from paddleocr import PaddleOCR

                self._ocr = PaddleOCR(use_angle_cls=True, lang="ch")
            except Exception as exc:
                self._ocr = None
                self.init_error = str(exc)
        else:
            self.init_error = "PaddleOCR disabled"

    def extract_text(self, file_path: str) -> Dict[str, object]:
        if self._ocr is None:
            return {
                "text": "",
                "pages": [],
                "warning": f"OCR engine unavailable: {getattr(self, 'init_error', 'unknown')}",
                "next_step": "Install PaddleOCR in a later phase if image OCR is required.",
            }
        result = self._ocr.ocr(file_path, cls=True)
        lines: List[str] = []
        line_items: List[Dict[str, object]] = []
        for page in result or []:
            for item in page or []:
                if len(item) >= 2 and item[1]:
                    text = str(item[1][0])
                    score = float(item[1][1]) if len(item[1]) > 1 else None
                    lines.append(text)
                    line_items.append({"text": text, "confidence": score, "box": item[0] if item else None})
        return {"text": "\n".join(lines), "pages": result or [], "lines": line_items}
