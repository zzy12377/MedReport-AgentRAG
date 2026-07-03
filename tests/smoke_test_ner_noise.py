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

from engines.ner.medical_ner import extract_medical_entities


def main() -> int:
    text = (
        "血 压150/95 mmHg  ALT85.2U/L  GLU 7.2mmol/l  LDL C 4.1 mmol/L "
        "WBC11.2 x10^9/L HGB 108g/L 心 率108次/分 BMI27.4kg/m2"
    )
    entities = extract_medical_entities(text)
    names = {row["name"] for row in entities}
    print(json.dumps(entities, ensure_ascii=False, indent=2))
    required = {"收缩压", "舒张压", "ALT", "GLU", "LDL-C", "WBC", "HGB", "心率", "BMI"}
    missing = required - names
    if missing:
        print(f"[ERROR] noisy NER missing required indicators: {sorted(missing)}")
        return 1
    print("[OK] smoke_test_ner_noise passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
