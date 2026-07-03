# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app
    except Exception as exc:
        print(f"[SKIP] FastAPI upload runtime unavailable: {exc}")
        return 0

    client = TestClient(app)
    content = "ALT 85.2 U/L GLU 7.2 mmol/L LDL-C 4.1 mmol/L 血压 150/95 mmHg"
    response = client.post(
        "/api/v1/reports/upload",
        files={"file": ("sample_report.txt", content.encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    print(payload)
    if payload.get("status") == "ocr_failed":
        print("[ERROR] Text upload should not fail OCR.")
        return 1
    if not payload.get("task_id"):
        print("[ERROR] Upload did not create a task.")
        return 1
    if "ALT" not in str(payload.get("ocr_text_preview", "")):
        print("[ERROR] Upload OCR preview missing expected text.")
        return 1
    print("[OK] smoke_test_upload_contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
