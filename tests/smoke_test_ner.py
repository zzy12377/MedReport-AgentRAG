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
    text = "ALT 85.2 U/L 参考范围 7-40；GLU 7.2 mmol/L 参考范围 3.9-6.1；LDL-C 4.1 mmol/L 参考范围 0-3.4"
    entities = extract_medical_entities(text)
    names = {e["name"] for e in entities}
    print(json.dumps(entities, ensure_ascii=False, indent=2))
    required = {"ALT", "GLU", "LDL-C"}
    missing = required - names
    if missing:
        print(f"[ERROR] NER missing required indicators: {sorted(missing)}")
        return 1
    print("[OK] smoke_test_ner passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
