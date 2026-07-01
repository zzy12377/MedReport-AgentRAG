# -*- coding: utf-8 -*-
"""Rule-based medical indicator extraction for phase-1 smoke tests.

The extractor intentionally avoids heavyweight NLP dependencies. It scans OCR
or manually typed physical-exam text and returns normalized lab/vital entities.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


INDICATORS: Dict[str, Dict[str, Any]] = {
    "ALT": {"aliases": ["ALT", "谷丙转氨酶", "丙氨酸氨基转移酶"], "unit": "U/L", "ref": (7.0, 40.0)},
    "AST": {"aliases": ["AST", "谷草转氨酶", "天门冬氨酸氨基转移酶"], "unit": "U/L", "ref": (13.0, 35.0)},
    "GGT": {"aliases": ["GGT", "γ-GT", "gamma-GT", "谷氨酰转肽酶"], "unit": "U/L", "ref": (7.0, 45.0)},
    "TBIL": {"aliases": ["TBIL", "总胆红素"], "unit": "umol/L", "ref": (3.4, 20.5)},
    "GLU": {"aliases": ["GLU", "FPG", "空腹血糖", "血糖", "葡萄糖"], "unit": "mmol/L", "ref": (3.9, 6.1)},
    "HbA1c": {"aliases": ["HbA1c", "糖化血红蛋白"], "unit": "%", "ref": (4.0, 6.0)},
    "TC": {"aliases": ["TC", "总胆固醇"], "unit": "mmol/L", "ref": (0.0, 5.2)},
    "TG": {"aliases": ["TG", "甘油三酯"], "unit": "mmol/L", "ref": (0.0, 1.7)},
    "LDL-C": {"aliases": ["LDL-C", "LDL", "低密度脂蛋白胆固醇"], "unit": "mmol/L", "ref": (0.0, 3.4)},
    "HDL-C": {"aliases": ["HDL-C", "HDL", "高密度脂蛋白胆固醇"], "unit": "mmol/L", "ref": (1.0, 2.1)},
    "Cr": {"aliases": ["Cr", "CREA", "肌酐"], "unit": "umol/L", "ref": (57.0, 111.0)},
    "尿酸": {"aliases": ["尿酸", "UA", "uric acid"], "unit": "umol/L", "ref": (208.0, 428.0)},
    "收缩压": {"aliases": ["收缩压", "SBP"], "unit": "mmHg", "ref": (90.0, 140.0)},
    "舒张压": {"aliases": ["舒张压", "DBP"], "unit": "mmHg", "ref": (60.0, 90.0)},
}

UNIT_PATTERN = r"(U/L|IU/L|mmol/L|μmol/L|umol/L|µmol/L|mg/dL|mmHg|%)"
NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?"


def _split_segments(text: str) -> List[str]:
    parts = re.split(r"[\n\r;；,，]+", str(text or ""))
    return [p.strip() for p in parts if p and p.strip()]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias)
    if re.fullmatch(r"[A-Za-z0-9%-]+", alias):
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)
    return re.compile(escaped, re.IGNORECASE)


def _find_value_after_alias(segment: str, alias: str) -> Optional[float]:
    match = _alias_pattern(alias).search(segment)
    if not match:
        return None
    tail = segment[match.end():]
    value_match = re.search(NUMBER_PATTERN, tail)
    if not value_match:
        return None
    try:
        return float(value_match.group(0))
    except ValueError:
        return None


def _extract_unit(segment: str, value: float, default_unit: str) -> str:
    value_text = re.escape(str(int(value)) if float(value).is_integer() else str(value))
    match = re.search(rf"{value_text}\s*{UNIT_PATTERN}", segment, flags=re.IGNORECASE)
    if match:
        unit = match.group(1)
        return "umol/L" if unit in {"μmol/L", "µmol/L"} else unit
    unit_match = re.search(UNIT_PATTERN, segment, flags=re.IGNORECASE)
    if unit_match:
        unit = unit_match.group(1)
        return "umol/L" if unit in {"μmol/L", "µmol/L"} else unit
    return default_unit


def _extract_ref_range(segment: str) -> Tuple[Optional[float], Optional[float]]:
    patterns = [
        rf"(?:参考范围|参考值|正常范围|正常值|ref(?:erence)?(?: range)?)\s*[:：]?\s*({NUMBER_PATTERN})\s*(?:-|~|至|到)\s*({NUMBER_PATTERN})",
        rf"\(({NUMBER_PATTERN})\s*(?:-|~|至|到)\s*({NUMBER_PATTERN})\)",
        rf"\b({NUMBER_PATTERN})\s*(?:-|~|至|到)\s*({NUMBER_PATTERN})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, segment, flags=re.IGNORECASE):
            low, high = float(match.group(1)), float(match.group(2))
            if low <= high:
                return low, high
    return None, None


def _entity(
    name: str,
    value: float,
    unit: str,
    ref_low: Optional[float],
    ref_high: Optional[float],
    original_text: str,
) -> Dict[str, Any]:
    is_abnormal = False
    if ref_low is not None and value < ref_low:
        is_abnormal = True
    if ref_high is not None and value > ref_high:
        is_abnormal = True
    return {
        "name": name,
        "value": value,
        "unit": unit,
        "ref_low": ref_low,
        "ref_high": ref_high,
        "is_abnormal": is_abnormal,
        "original_text": original_text.strip(),
    }


def _extract_blood_pressure(text: str) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    bp_match = re.search(r"(?:血压|BP)\s*[:：]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmHg)?", text, re.IGNORECASE)
    if bp_match:
        sbp = float(bp_match.group(1))
        dbp = float(bp_match.group(2))
        original = bp_match.group(0)
        entities.append(_entity("收缩压", sbp, "mmHg", 90.0, 140.0, original))
        entities.append(_entity("舒张压", dbp, "mmHg", 60.0, 90.0, original))

    for name in ["收缩压", "舒张压"]:
        spec = INDICATORS[name]
        for alias in spec["aliases"]:
            for segment in _split_segments(text):
                value = _find_value_after_alias(segment, alias)
                if value is None:
                    continue
                low, high = _extract_ref_range(segment)
                if low is None:
                    low, high = spec["ref"]
                entities.append(_entity(name, value, "mmHg", low, high, segment))
    return entities


def extract_medical_entities(text: str) -> List[Dict[str, Any]]:
    """Extract normalized physical-exam indicators from free text."""
    entities: List[Dict[str, Any]] = []
    seen = set()

    for bp_entity in _extract_blood_pressure(text):
        key = (bp_entity["name"], bp_entity["original_text"])
        if key not in seen:
            seen.add(key)
            entities.append(bp_entity)

    segments = _split_segments(text)
    indicator_items = sorted(
        INDICATORS.items(),
        key=lambda item: max(len(alias) for alias in item[1]["aliases"]),
        reverse=True,
    )

    for segment in segments:
        for name, spec in indicator_items:
            if name in {"收缩压", "舒张压"}:
                continue
            for alias in spec["aliases"]:
                value = _find_value_after_alias(segment, alias)
                if value is None:
                    continue
                low, high = _extract_ref_range(segment)
                if low is None:
                    low, high = spec["ref"]
                unit = _extract_unit(segment, value, spec["unit"])
                key = (name, segment)
                if key not in seen:
                    seen.add(key)
                    entities.append(_entity(name, value, unit, low, high, segment))
                break

    return entities


def entities_to_query_text(entities: Iterable[Dict[str, Any]]) -> str:
    """Convert extracted entities into a compact retrieval query."""
    parts = []
    for entity in entities:
        name = entity.get("name", "")
        value = entity.get("value", "")
        unit = entity.get("unit", "")
        status = "abnormal" if entity.get("is_abnormal") else "normal"
        parts.append(f"{name}: {value} {unit} ({status})")
    return "; ".join(parts)


if __name__ == "__main__":
    sample = "ALT 85.2 U/L 参考范围 7-40；GLU 7.2 mmol/L 参考范围 3.9-6.1；LDL-C 4.1 mmol/L 参考范围 0-3.4；血压 150/95 mmHg"
    import json

    print(json.dumps(extract_medical_entities(sample), ensure_ascii=False, indent=2))

