# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engines.kg.chinese_kg import ChineseMedicalKGRetriever


def main() -> int:
    entities = [
        {"name": "ALT", "value": 85.2, "unit": "U/L", "ref_low": 7, "ref_high": 40, "is_abnormal": True},
        {"name": "LDL-C", "value": 4.1, "unit": "mmol/L", "ref_low": 0, "ref_high": 3.4, "is_abnormal": True},
        {"name": "GLU", "value": 7.2, "unit": "mmol/L", "ref_low": 3.9, "ref_high": 6.1, "is_abnormal": True},
    ]
    rows = ChineseMedicalKGRetriever(os.path.join(ROOT, "resources", "chinese_medical_kg.json")).retrieve(entities, top_k=8)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    tails = " ".join(str(row.get("tail", "")) for row in rows)
    required = ["肝", "血脂", "糖"]
    missing = [word for word in required if word not in tails]
    if missing:
        print(f"[ERROR] Chinese KG missing expected evidence terms: {missing}")
        return 1
    print("[OK] smoke_test_chinese_kg passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
