# -*- coding: utf-8 -*-
"""Runtime entity extractor adapter."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from engines.ner.medical_ner import extract_medical_entities


class EntityExtractor:
    """Extract coarse patient features and lab indicators from free text."""

    def extract(self, text: str) -> Dict[str, Any]:
        raw = str(text or "")
        indicators = extract_medical_entities(raw)
        return {
            "age": self._extract_age(raw),
            "sex": self._extract_sex(raw),
            "symptoms": raw,
            "location": "",
            "restriction": "",
            "indicators": indicators,
        }

    @staticmethod
    def _extract_age(text: str) -> int | None:
        patterns = [
            r"(\d{1,3})\s*(?:year-old|years old|yo)\b",
            r"(?:age|aged)\s*[:：]?\s*(\d{1,3})\b",
            r"(\d{1,3})\s*岁",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                age = int(match.group(1))
                if 0 < age < 130:
                    return age
        return None

    @staticmethod
    def _extract_sex(text: str) -> str | None:
        if re.search(r"\b(male|man|boy|m)\b|男", text, flags=re.IGNORECASE):
            return "M"
        if re.search(r"\b(female|woman|girl|f)\b|女", text, flags=re.IGNORECASE):
            return "F"
        return None


def extract_features(text: str) -> Dict[str, Any]:
    return EntityExtractor().extract(text)
