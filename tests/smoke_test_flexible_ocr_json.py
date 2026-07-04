# -*- coding: utf-8 -*-
"""Smoke test for flexible structured OCR JSON normalization."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.services.ocr_json_service import normalize_ocr_json


def main() -> int:
    payload = {
        "name": "张三",
        "gender": "男",
        "age": 28,
        "height": {"value": 175.0, "unit": "cm"},
        "weight": {"value": 68.0, "unit": "kg"},
        "blood_pressure": {"systolic": 118, "diastolic": 76, "unit": "mmHg"},
        "heart_rate": {"value": 72, "unit": "bpm"},
        "conclusion": "各项指标基本正常，建议保持规律作息、均衡饮食、适量运动，每年定期体检。",
        "_source": "local_rule_parser",
        "_text_length": 105,
    }
    normalized = normalize_ocr_json(payload)
    text = normalized["text"]
    required = ["姓名 张三", "性别 男", "年龄 28岁", "身高 175.0 cm", "体重 68.0 kg", "血压 118/76 mmHg", "心率 72 bpm"]
    missing = [item for item in required if item not in text]
    assert not missing, f"missing normalized fragments: {missing}\n{text}"
    assert "各项指标基本正常" not in text
    notes = normalized.get("interpretive_notes") or []
    assert notes and "各项指标基本正常" in notes[0].get("text", "")
    assert "_source" not in text
    print("[OK] flexible OCR JSON normalization passed.")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
