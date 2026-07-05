# -*- coding: utf-8 -*-
"""Smoke test for OCR JSON -> diagnosis API flow."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    os.environ.setdefault("FORCE_MOCK_LLM", "true")
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app
    except Exception as exc:
        print(f"[SKIP] FastAPI runtime unavailable: {exc}")
        print("Next step: pip install fastapi uvicorn python-multipart")
        return 0

    client = TestClient(app)
    payload = {
        "case_id": "ocr-smoke-001",
        "top_k": 1,
        "use_multi_agent": False,
        "use_kg": False,
        "ocr_json": {
            "pages": [
                {
                    "page_no": 1,
                    "lines": [
                        {"text": "ALT 85.2 U/L 参考范围 7-40"},
                        {"text": "GLU 7.2 mmol/L 参考范围 3.9-6.1"},
                        {"text": "LDL-C 4.1 mmol/L 参考范围 0-3.4"},
                        {"text": "血压 150/95 mmHg"},
                    ],
                }
            ]
        },
    }
    response = client.post("/api/v1/diagnosis/ocr-json/sync", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body.get("status") == "done", body
    assert body.get("input_type") == "ocr_json", body
    assert body.get("task_id"), body
    assert body.get("report_id") == body.get("task_id"), body
    assert body.get("report_path"), body
    normalized = body.get("normalized_input") or {}
    assert "ALT 85.2" in normalized.get("text", ""), normalized
    assert normalized.get("case_id") == "ocr-smoke-001", normalized
    report = body.get("report") or {}
    for key in ["task_id", "retrieved_cases", "kg_evidence", "entities", "summary_markdown", "safety_note"]:
        assert key in report, f"missing report field: {key}"
    assert report.get("case_id") == "ocr-smoke-001", report
    alias = client.post("/api/v1/reports/from-ocr-json", json={"plain_text": "ALT 85.2 U/L\nGLU 7.2 mmol/L"})
    assert alias.status_code == 200, alias.text
    assert alias.json().get("report_id"), alias.text

    simple = client.post("/api/v1/reports/from-ocr-json/simple", json={"plain_text": "ALT 85.2 U/L\nGLU 7.2 mmol/L"})
    assert simple.status_code == 200, simple.text
    simple_body = simple.json()
    assert set(simple_body).issuperset({"status", "task_id", "report_text", "format"}), simple_body
    assert simple_body["format"] == "markdown", simple_body
    assert "report" not in simple_body, simple_body
    assert "检测" in simple_body["report_text"] or "Diagnosis" in simple_body["report_text"], simple_body

    markdown = client.post("/api/v1/reports/from-ocr-json/markdown", json={"plain_text": "ALT 85.2 U/L\nGLU 7.2 mmol/L"})
    assert markdown.status_code == 200, markdown.text
    assert markdown.headers["content-type"].startswith("text/plain"), markdown.headers
    assert "ALT" in markdown.text or "检测" in markdown.text or "Diagnosis" in markdown.text, markdown.text
    print("[OK] OCR JSON API smoke test passed.")
    print(f"[INFO] normalized_lines={normalized.get('line_count')}")
    print(f"[INFO] entities={len(report.get('entities') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
