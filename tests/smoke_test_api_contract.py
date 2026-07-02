# -*- coding: utf-8 -*-
"""Smoke test for the document-aligned FastAPI contract."""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def main() -> int:
    try:
        from fastapi.testclient import TestClient
        from backend.app.main import app
    except Exception as exc:
        print(f"[SKIP] FastAPI runtime unavailable: {exc}")
        print("Next step: pip install fastapi uvicorn python-multipart")
        return 0

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200, health.text

    response = client.post(
        "/api/v1/diagnosis/text/sync",
        json={
            "text": "Patient male 56 years old BP 150/95 mmHg LDL-C 4.2 mmol/L GLU 7.1 mmol/L ALT 68 U/L",
            "top_k": 1,
            "use_multi_agent": True,
            "vector_sources": ["all"],
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("status") == "done", payload
    report = payload.get("report") or {}
    required = [
        "task_id",
        "overall_risk",
        "possible_diagnoses",
        "retrieved_cases",
        "kg_evidence",
        "agent_opinions",
        "critique",
        "summary_markdown",
        "entities",
        "safety_note",
    ]
    missing = [key for key in required if key not in report]
    assert not missing, f"missing report fields: {missing}"
    print("[OK] FastAPI contract smoke test passed.")
    print(f"[INFO] retrieved_cases={len(report.get('retrieved_cases') or [])}")
    print(f"[INFO] kg_evidence={len(report.get('kg_evidence') or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
