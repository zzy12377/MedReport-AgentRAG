# -*- coding: utf-8 -*-
"""Rule-based medical indicator extraction for phase-1 smoke tests.

The extractor intentionally avoids heavyweight NLP dependencies. It scans OCR
or manually typed physical-exam text and returns normalized lab/vital entities.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


INDICATORS: Dict[str, Dict[str, Any]] = {
    # Liver function
    "ALT": {"aliases": ["ALT", "GPT", "谷丙转氨酶", "丙氨酸氨基转移酶"], "unit": "U/L", "ref": (7.0, 40.0)},
    "AST": {"aliases": ["AST", "GOT", "谷草转氨酶", "天门冬氨酸氨基转移酶"], "unit": "U/L", "ref": (13.0, 35.0)},
    "GGT": {"aliases": ["GGT", "G-GT", "γ-GT", "gamma-GT", "谷氨酰转肽酶"], "unit": "U/L", "ref": (7.0, 45.0)},
    "ALP": {"aliases": ["ALP", "碱性磷酸酶"], "unit": "U/L", "ref": (45.0, 125.0)},
    "TBIL": {"aliases": ["TBIL", "总胆红素"], "unit": "umol/L", "ref": (3.4, 20.5)},
    "DBIL": {"aliases": ["DBIL", "直接胆红素"], "unit": "umol/L", "ref": (0.0, 6.8)},
    "IBIL": {"aliases": ["IBIL", "间接胆红素"], "unit": "umol/L", "ref": (1.7, 13.7)},
    "ALB": {"aliases": ["ALB", "白蛋白"], "unit": "g/L", "ref": (40.0, 55.0)},
    # Glucose and endocrine
    "GLU": {"aliases": ["GLU", "FPG", "空腹血糖", "血糖", "葡萄糖"], "unit": "mmol/L", "ref": (3.9, 6.1)},
    "HbA1c": {"aliases": ["HbA1c", "HbAlc", "糖化血红蛋白"], "unit": "%", "ref": (4.0, 6.0)},
    "TSH": {"aliases": ["TSH", "促甲状腺激素"], "unit": "mIU/L", "ref": (0.27, 4.2)},
    "FT3": {"aliases": ["FT3", "游离三碘甲状腺原氨酸"], "unit": "pmol/L", "ref": (3.1, 6.8)},
    "FT4": {"aliases": ["FT4", "游离甲状腺素"], "unit": "pmol/L", "ref": (12.0, 22.0)},
    # Lipids
    "TC": {"aliases": ["TC", "CHOL", "总胆固醇", "胆固醇"], "unit": "mmol/L", "ref": (0.0, 5.2)},
    "TG": {"aliases": ["TG", "甘油三酯"], "unit": "mmol/L", "ref": (0.0, 1.7)},
    "LDL-C": {"aliases": ["LDL-C", "LDL C", "LDLC", "LDL", "低密度脂蛋白胆固醇", "低密度脂蛋白"], "unit": "mmol/L", "ref": (0.0, 3.4)},
    "HDL-C": {"aliases": ["HDL-C", "HDL C", "HDLC", "HDL", "高密度脂蛋白胆固醇", "高密度脂蛋白"], "unit": "mmol/L", "ref": (1.0, 2.1)},
    # Kidney and metabolism
    "Cr": {"aliases": ["Cr", "CREA", "Scr", "肌酐", "血肌酐"], "unit": "umol/L", "ref": (57.0, 111.0)},
    "BUN": {"aliases": ["BUN", "尿素氮", "血尿素氮"], "unit": "mmol/L", "ref": (2.9, 8.2)},
    "尿酸": {"aliases": ["尿酸", "UA", "uric acid", "血尿酸"], "unit": "umol/L", "ref": (208.0, 428.0)},
    # Blood routine
    "WBC": {"aliases": ["WBC", "白细胞", "白细胞计数"], "unit": "10^9/L", "ref": (3.5, 9.5)},
    "RBC": {"aliases": ["RBC", "红细胞", "红细胞计数"], "unit": "10^12/L", "ref": (3.8, 5.8)},
    "HGB": {"aliases": ["HGB", "Hb", "血红蛋白"], "unit": "g/L", "ref": (115.0, 150.0)},
    "PLT": {"aliases": ["PLT", "血小板", "血小板计数"], "unit": "10^9/L", "ref": (125.0, 350.0)},
    # Vitals and body measurements
    "收缩压": {"aliases": ["收缩压", "SBP"], "unit": "mmHg", "ref": (90.0, 140.0)},
    "舒张压": {"aliases": ["舒张压", "DBP"], "unit": "mmHg", "ref": (60.0, 90.0)},
    "BMI": {"aliases": ["BMI", "体质指数", "身体质量指数"], "unit": "kg/m2", "ref": (18.5, 24.0)},
    "心率": {"aliases": ["心率", "HR", "脉搏"], "unit": "bpm", "ref": (60.0, 100.0)},
}

UNIT_PATTERN = (
    r"(10\^12/L|x10\^12/L|×10\^12/L|10\*12/L|10\^9/L|x10\^9/L|×10\^9/L|10\*9/L|"
    r"mIU/L|pmol/L|U/L|IU/L|mmol/L|μmol/L|umol/L|µmol/L|mg/dL|g/dL|g/L|mmHg|kg/m2|kg/m²|"
    r"bpm|次/分|%)"
)
NUMBER_PATTERN = r"[-+]?\d+(?:\.\d+)?"


def _normalize_text(text: str) -> str:
    """Normalize common OCR and copy-paste variants before rule matching."""
    normalized = str(text or "")
    replacements = {
        "，": ",",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "－": "-",
        "—": "-",
        "–": "-",
        "～": "~",
        "％": "%",
        "／": "/",
        "×": "x",
        "μ": "u",
        "µ": "u",
        "㎜": "mm",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    normalized = re.sub(r"(?i)\b(u)\s*mol\s*/\s*l\b", "umol/L", normalized)
    normalized = re.sub(r"(?i)\b(mmol|pmol|miu|iu|u|g|mg)\s*/\s*l\b", lambda m: f"{m.group(1)}/L", normalized)
    normalized = re.sub(r"(?i)\bmm\s*hg\b", "mmHg", normalized)
    normalized = re.sub(r"(?i)\bkg\s*/\s*m\s*(?:2|\^2|²)\b", "kg/m2", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _split_segments(text: str) -> List[str]:
    parts = re.split(r"[\n\r;,]+", _normalize_text(text))
    return [p.strip() for p in parts if p and p.strip()]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    alias = _normalize_text(alias)
    if any("\u4e00" <= ch <= "\u9fff" for ch in alias):
        body = r"\s*".join(re.escape(ch) for ch in alias)
        return re.compile(body, re.IGNORECASE)
    if re.fullmatch(r"[A-Za-z0-9% -]+", alias):
        chunks = []
        for ch in alias:
            if ch in {" ", "-"}:
                chunks.append(r"[-\s]?")
            else:
                chunks.append(re.escape(ch) + r"\s*")
        body = "".join(chunks).rstrip(r"\s*")
        # Values are often pasted directly after aliases, e.g. ALT85.2U/L.
        # Keep the left boundary strict, but allow a digit on the right.
        return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z])", re.IGNORECASE)
    return re.compile(re.escape(alias), re.IGNORECASE)


def _find_value_after_alias(segment: str, alias: str) -> Optional[float]:
    result = _find_value_context_after_alias(segment, alias)
    return result[0] if result else None


def _find_value_context_after_alias(segment: str, alias: str) -> Optional[Tuple[float, str]]:
    match = _alias_pattern(alias).search(segment)
    if not match:
        return None
    prefix = segment[max(0, match.start() - 4):match.start()]
    if alias == "血红蛋白" and "糖化" in prefix:
        return None
    tail = segment[match.end():]
    value_match = re.search(NUMBER_PATTERN, tail)
    if not value_match:
        return None
    try:
        value = float(value_match.group(0))
        return value, tail[value_match.start():]
    except ValueError:
        return None


def _extract_unit(segment: str, value: float, default_unit: str) -> str:
    value_text = re.escape(str(int(value)) if float(value).is_integer() else str(value))
    match = re.search(rf"{value_text}\s*{UNIT_PATTERN}", segment, flags=re.IGNORECASE)
    if match:
        return _canonical_unit(match.group(1))
    unit_match = re.search(UNIT_PATTERN, segment, flags=re.IGNORECASE)
    if unit_match:
        return _canonical_unit(unit_match.group(1))
    return default_unit


def _canonical_unit(unit: str) -> str:
    compact = _normalize_text(unit).replace(" ", "")
    lower = compact.lower()
    unit_map = {
        "u/l": "U/L",
        "iu/l": "IU/L",
        "mmol/l": "mmol/L",
        "umol/l": "umol/L",
        "pmol/l": "pmol/L",
        "miu/l": "mIU/L",
        "mg/dl": "mg/dL",
        "g/dl": "g/dL",
        "g/l": "g/L",
        "mmhg": "mmHg",
        "kg/m2": "kg/m2",
        "kg/m²": "kg/m2",
        "bpm": "bpm",
        "次/分": "bpm",
        "%": "%",
    }
    if lower in {"x10^9/l", "10*9/l", "10^9/l"}:
        return "10^9/L"
    if lower in {"x10^12/l", "10*12/l", "10^12/l"}:
        return "10^12/L"
    return unit_map.get(lower, compact)


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
    normalized = _normalize_text(text)
    for bp_match in re.finditer(r"(?:血压|BP|blood pressure)\s*[:]?\s*(\d{2,3})\s*/\s*(\d{2,3})\s*(?:mmHg)?", normalized, re.IGNORECASE):
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
                value_context = _find_value_context_after_alias(segment, alias)
                if value_context is None:
                    continue
                value, context = value_context
                low, high = _extract_ref_range(segment)
                if low is None:
                    low, high = spec["ref"]
                unit = _extract_unit(context, value, spec["unit"])
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
