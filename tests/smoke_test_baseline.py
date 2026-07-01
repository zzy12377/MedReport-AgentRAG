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

from baselines.run_b0_direct import run_b0
from baselines.run_b1_rag import run_b1


def _check_standard(result: dict) -> bool:
    required = {
        "case_id",
        "mode",
        "prediction",
        "ground_truth",
        "l1_pred",
        "l2_pred",
        "l3_pred",
        "l1_truth",
        "l2_truth",
        "l3_truth",
        "retrieved_cases",
        "kg_evidence",
        "agent_outputs",
        "critique",
        "safety_note",
    }
    missing = required - set(result)
    if missing:
        print(f"[ERROR] Missing standard keys: {sorted(missing)}")
        return False
    return True


def main() -> int:
    text = "ALT 85.2 U/L 参考范围 7-40；GLU 7.2 mmol/L；LDL-C 4.1 mmol/L；血压 150/95 mmHg"
    b0 = run_b0(text, case_id="smoke", mock=True, output_dir=None)
    b1 = run_b1(text, case_id="smoke", mock=True, top_k=1, output_dir=None)
    print(json.dumps({"B0": b0, "B1": b1}, ensure_ascii=False, indent=2)[:5000])
    if not (_check_standard(b0) and _check_standard(b1)):
        return 1
    print("[OK] smoke_test_baseline passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
