# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import tempfile

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.app.services.ocr_service import OCRService


def main() -> int:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write("ALT 85.2 U/L GLU 7.2 mmol/L LDL-C 4.1 mmol/L")
        path = f.name
    try:
        result = OCRService().extract(path)
        print(result)
        if "ALT" not in str(result.get("text", "")):
            print("[ERROR] OCRService failed to read text files.")
            return 1
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    print("[OK] smoke_test_ocr_service passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
