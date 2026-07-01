# -*- coding: utf-8 -*-
"""PDF text parser with graceful optional dependency handling."""

from __future__ import annotations

from typing import Dict


def extract_pdf_text(file_path: str) -> Dict[str, object]:
    try:
        import pdfplumber
    except Exception as exc:
        return {
            "text": "",
            "pages": [],
            "warning": f"pdfplumber unavailable: {exc}",
            "next_step": "Install pdfplumber or route scanned PDFs through OCR.",
        }

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": idx, "text": text})
    return {"text": "\n".join(p["text"] for p in pages), "pages": pages}

