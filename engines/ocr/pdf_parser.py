# -*- coding: utf-8 -*-
"""PDF text parser with graceful optional dependency handling."""

from __future__ import annotations

from typing import Dict, List


def _extract_with_pymupdf(file_path: str) -> Dict[str, object]:
    import fitz

    pages: List[Dict[str, object]] = []
    with fitz.open(file_path) as doc:
        for idx, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            pages.append({"page": idx, "text": text})
    return {"text": "\n".join(str(p["text"]) for p in pages).strip(), "pages": pages, "parser": "pymupdf"}


def _extract_with_pdfplumber(file_path: str) -> Dict[str, object]:
    import pdfplumber

    pages = []
    with pdfplumber.open(file_path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": idx, "text": text})
    return {"text": "\n".join(str(p["text"]) for p in pages).strip(), "pages": pages, "parser": "pdfplumber"}


def extract_pdf_text(file_path: str) -> Dict[str, object]:
    errors = []
    try:
        result = _extract_with_pymupdf(file_path)
        if str(result.get("text") or "").strip():
            return result
        errors.append("PyMuPDF returned empty text")
    except Exception as exc:
        errors.append(f"PyMuPDF unavailable or failed: {exc}")

    try:
        result = _extract_with_pdfplumber(file_path)
        if str(result.get("text") or "").strip():
            return result
        errors.append("pdfplumber returned empty text")
    except Exception as exc:
        errors.append(f"pdfplumber unavailable or failed: {exc}")

    return {
        "text": "",
        "pages": [],
        "warning": "; ".join(errors) or "No text extracted from PDF.",
        "next_step": "For scanned PDFs, install PyMuPDF plus PaddleOCR or upload page images.",
    }


def render_pdf_pages_to_images(file_path: str, output_dir: str, dpi: int = 180) -> Dict[str, object]:
    """Render PDF pages to PNG files when PyMuPDF is available."""
    try:
        import fitz
    except Exception as exc:
        return {
            "image_paths": [],
            "warning": f"PyMuPDF unavailable: {exc}",
            "next_step": "Install pymupdf to OCR scanned PDF pages.",
        }
    import os

    os.makedirs(output_dir, exist_ok=True)
    image_paths: List[str] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(file_path) as doc:
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            path = os.path.join(output_dir, f"page_{idx:03d}.png")
            pix.save(path)
            image_paths.append(path)
    return {"image_paths": image_paths, "warning": ""}
