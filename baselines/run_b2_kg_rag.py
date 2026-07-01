# -*- coding: utf-8 -*-
"""B2 baseline skeleton: KG-RAG + multi-agent interface contract."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common import make_standard_result, read_text_input, save_result
from baselines.run_b1_rag import run_b1


def run_b2(
    case_text: str,
    case_id: str = "manual",
    ground_truth: str = "",
    mock: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    b1 = run_b1(case_text, case_id=case_id, ground_truth=ground_truth, mock=mock, output_dir=None)
    result = make_standard_result(
        case_id=case_id,
        mode="B2",
        prediction=b1.get("prediction", ""),
        ground_truth=ground_truth,
        retrieved_cases=b1.get("retrieved_cases", []),
        kg_evidence=[{"status": "TODO", "message": "KG subgraph extraction will be completed in phase 2."}],
        agent_outputs=[{"status": "TODO", "message": "Specialist agents will be completed in phase 2."}],
        critique={"status": "TODO", "message": "Critique calibration will be completed in phase 2."},
    )
    if output_dir:
        result["output_files"] = save_result(result, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B2 KG-RAG skeleton baseline.")
    parser.add_argument("--text", default="")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--case-file", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output-dir", default="./storage/results")
    args = parser.parse_args()
    payload = read_text_input(args.text, args.input_file, args.case_file)
    result = run_b2(payload["text"], payload["case_id"], payload["ground_truth"], mock=True, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
