# -*- coding: utf-8 -*-
"""OCR service wrapper.

Text files are read directly. Images/PDFs go through the optional OCR engine; if
PaddleOCR is unavailable the service returns an explicit placeholder message.
"""

from __future__ import annotations

import os
import uuid

from backend.app.core.ocr.paddle_ocr_engine import PaddleOCREngine
from backend.app.config.settings import settings
from engines.ocr.pdf_parser import extract_pdf_text, render_pdf_pages_to_images


class OCRService:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def extract_text(self, file_path: str) -> str:
        return str(self.extract(file_path).get("text") or "")

    def extract(self, file_path: str) -> dict:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {".txt", ".md", ".csv"}:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            return {"text": text, "source_type": "text", "warning": ""}

        if ext == ".pdf":
            parsed = extract_pdf_text(file_path)
            if str(parsed.get("text") or "").strip():
                parsed["source_type"] = "pdf_text"
                return parsed
            rendered = self._ocr_scanned_pdf(file_path)
            if str(rendered.get("text") or "").strip():
                return rendered
            warning = "; ".join(
                item
                for item in [str(parsed.get("warning") or ""), str(rendered.get("warning") or "")]
                if item
            )
            return {
                "text": "",
                "source_type": "pdf",
                "warning": warning or "No text extracted from PDF.",
                "next_step": "Install pdf text/OCR dependencies or upload a text file for demo.",
            }

        if ext in self.image_exts:
            return self._ocr_image(file_path)

        return {
            "text": "",
            "source_type": "unsupported",
            "warning": f"Unsupported file type: {ext or '(none)'}",
            "next_step": "Upload txt, md, csv, pdf, png, jpg, jpeg, bmp, tif, tiff, or webp.",
        }

    def _ocr_image(self, file_path: str) -> dict:
        result = PaddleOCREngine().extract_text(file_path)
        text = str(result.get("text") or "").strip()
        if text:
            result["source_type"] = "image_ocr"
            return result
        warning = result.get("warning") or "OCR engine unavailable."
        return {"text": "", "source_type": "image_ocr", "warning": warning, "next_step": result.get("next_step", "")}

    def _ocr_scanned_pdf(self, file_path: str) -> dict:
        page_dir = os.path.join(settings.upload_dir, "_pdf_pages", uuid.uuid4().hex)
        rendered = render_pdf_pages_to_images(file_path, page_dir)
        image_paths = list(rendered.get("image_paths") or [])
        if not image_paths:
            return {"text": "", "source_type": "pdf_ocr", "warning": rendered.get("warning", "")}
        texts = []
        warnings = []
        for image_path in image_paths:
            result = self._ocr_image(image_path)
            if result.get("text"):
                texts.append(str(result["text"]))
            if result.get("warning"):
                warnings.append(str(result["warning"]))
        return {
            "text": "\n".join(texts).strip(),
            "source_type": "pdf_ocr",
            "pages": [{"image_path": path} for path in image_paths],
            "warning": "; ".join(dict.fromkeys(warnings)),
        }

    def save_ocr_text(self, file_id: str, text: str) -> str:
        os.makedirs(settings.ocr_text_dir, exist_ok=True)
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(file_id))
        path = os.path.join(settings.ocr_text_dir, f"{safe_id}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        return path
