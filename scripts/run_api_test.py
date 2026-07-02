# -*- coding: utf-8 -*-
"""Small API smoke test for the FastAPI runtime."""

from __future__ import annotations

import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Call /api/v1/diagnosis/text/sync once.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/api/v1/diagnosis/text/sync")
    parser.add_argument(
        "--text",
        default="Patient male, 56 years old, BP 150/95 mmHg, LDL-C 4.2 mmol/L, GLU 7.1 mmol/L, ALT 68 U/L.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    import requests

    resp = requests.post(
        args.url,
        json={"text": args.text, "top_k": args.top_k, "use_multi_agent": True, "vector_sources": ["all"]},
        timeout=180,
    )
    print(f"HTTP {resp.status_code}")
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
    return 0 if resp.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
