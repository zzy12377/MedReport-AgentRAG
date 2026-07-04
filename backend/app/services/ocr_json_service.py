# -*- coding: utf-8 -*-
"""Normalize OCR JSON payloads into diagnosis-ready text."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


TEXT_KEYS = {
    "text",
    "ocr_text",
    "full_text",
    "recognized_text",
    "recognised_text",
    "line_text",
    "content",
    "value_text",
}

CONTAINER_KEYS = {
    "pages",
    "page",
    "lines",
    "line",
    "blocks",
    "paragraphs",
    "results",
    "result",
    "ocr_result",
    "ocr_results",
    "data",
    "items",
}


def normalize_ocr_json(payload: Any) -> Dict[str, Any]:
    """Return a stable text representation from common OCR JSON shapes.

    The function is intentionally permissive because OCR vendors and frontend
    demos often produce slightly different JSON. It prefers explicit text
    fields, then recursively extracts text-like values from page/line/result
    containers, and finally synthesizes lines from structured indicator items.
    """

    fragments: List[str] = []
    fragments.extend(_extract_indicator_lines(payload))
    fragments.extend(_extract_text_fragments(payload))
    fragments = _dedupe_preserve_order(_clean_fragment(item) for item in fragments)
    text = "\n".join(item for item in fragments if item)
    return {
        "text": text,
        "line_count": len([item for item in fragments if item]),
        "source_format": _guess_source_format(payload),
        "raw": payload,
    }


def build_diagnosis_text_from_ocr_json(payload: Any) -> str:
    normalized = normalize_ocr_json(payload)
    text = str(normalized.get("text") or "").strip()
    if not text:
        raise ValueError(
            "OCR JSON 中没有可用于诊断的文本。请确认 JSON 包含 text、ocr_text、pages、lines、blocks 或 results 等字段。"
        )
    return text


def split_ocr_request_payload(payload: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Accept wrapped or raw OCR JSON request bodies.

    Wrapped shape:
        {"ocr_json": {...}, "top_k": 5, "use_kg": true}

    Raw shape:
        {"pages": [{"lines": [{"text": "..."}]}]}
    """

    option_keys = {"case_id", "top_k", "use_multi_agent", "use_kg", "vector_sources"}
    if "ocr_json" in payload:
        options = {key: payload.get(key) for key in option_keys if key in payload}
        return payload.get("ocr_json"), options
    if "ocr_result" in payload and any(key in payload for key in option_keys):
        options = {key: payload.get(key) for key in option_keys if key in payload}
        return payload.get("ocr_result"), options
    return payload, {}


def _extract_text_fragments(value: Any) -> List[str]:
    fragments: List[str] = []
    if value is None:
        return fragments
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            fragments.append(stripped)
        return fragments
    if isinstance(value, (int, float, bool)):
        return fragments
    if isinstance(value, dict):
        for key, item in value.items():
            key_l = str(key).lower()
            if key_l in TEXT_KEYS and isinstance(item, str):
                fragments.append(item)
            elif key_l in CONTAINER_KEYS or isinstance(item, (dict, list, tuple)):
                fragments.extend(_extract_text_fragments(item))
        return fragments
    if isinstance(value, (list, tuple)):
        paddle_text = _extract_paddle_tuple_text(value)
        if paddle_text:
            fragments.append(paddle_text)
            return fragments
        for item in value:
            fragments.extend(_extract_text_fragments(item))
    return fragments


def _extract_paddle_tuple_text(value: Iterable[Any]) -> str:
    """Handle PaddleOCR-like rows: [box, [text, score]] or [box, (text, score)]."""

    items = list(value)
    if len(items) >= 2 and isinstance(items[1], (list, tuple)) and items[1]:
        first = items[1][0]
        if isinstance(first, str):
            return first
    return ""


def _extract_indicator_lines(value: Any) -> List[str]:
    lines: List[str] = []
    if isinstance(value, dict):
        for key in ("indicators", "entities", "medical_entities", "items"):
            items = value.get(key)
            if isinstance(items, list):
                for item in items:
                    line = _indicator_to_line(item)
                    if line:
                        lines.append(line)
        for item in value.values():
            if isinstance(item, (dict, list)):
                lines.extend(_extract_indicator_lines(item))
    elif isinstance(value, list):
        for item in value:
            lines.extend(_extract_indicator_lines(item))
    return lines


def _indicator_to_line(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    name = str(item.get("name") or item.get("label") or "").strip()
    value = item.get("value")
    unit = str(item.get("unit") or "").strip()
    if not name or value in (None, ""):
        return ""
    ref_low = item.get("ref_low")
    ref_high = item.get("ref_high")
    reference = item.get("reference_range") or item.get("ref_range")
    line = f"{name} {value} {unit}".strip()
    if ref_low not in (None, "") and ref_high not in (None, ""):
        line = f"{line} 参考范围 {ref_low}-{ref_high}"
    elif reference:
        line = f"{line} 参考范围 {reference}"
    return line


def _clean_fragment(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(part.strip() for part in text.split("\n") if part.strip())


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _guess_source_format(payload: Any) -> str:
    if isinstance(payload, list):
        return "list"
    if not isinstance(payload, dict):
        return type(payload).__name__
    keys = {str(key).lower() for key in payload.keys()}
    if "ocr_json" in keys:
        return "wrapped_ocr_json"
    if keys & {"pages", "lines", "blocks"}:
        return "document_ocr_json"
    if keys & {"indicators", "entities", "medical_entities"}:
        return "structured_medical_json"
    if keys & TEXT_KEYS:
        return "text_json"
    return "generic_json"
