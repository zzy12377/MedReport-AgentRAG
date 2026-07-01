# -*- coding: utf-8 -*-
"""B0 baseline: direct prompting without RAG or KG."""

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
from engines.llm.llm_gateway import LLMGateway, extract_prediction


SYSTEM_PROMPT = (
    "You are a cautious medical decision-support assistant. "
    "Analyze the provided case text and output a concise differential diagnosis. "
    "This is for course demonstration only."
)


def run_b0(
    case_text: str,
    case_id: str = "manual",
    ground_truth: str = "",
    mock: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = f"Case text:\n{case_text}\n\nPlease provide the most likely diagnosis and brief reasoning."
    response = LLMGateway(mock=mock).generate(prompt, system_prompt=SYSTEM_PROMPT, mode="B0")
    result = make_standard_result(
        case_id=case_id,
        mode="B0",
        prediction=extract_prediction(response),
        ground_truth=ground_truth,
    )
    result["llm_response"] = response
    if output_dir:
        result["output_files"] = save_result(result, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B0 direct prompting baseline.")
    parser.add_argument("--text", default="", help="Manual case text.")
    parser.add_argument("--input-file", default=None, help="Plain text input file.")
    parser.add_argument("--case-file", default=None, help="DDXPlus participant JSON file.")
    parser.add_argument("--mock", action="store_true", help="Force mock LLM output.")
    parser.add_argument("--output-dir", default="./storage/results", help="Where JSON/CSV outputs are saved.")
    args = parser.parse_args()

    payload = read_text_input(args.text, args.input_file, args.case_file)
    if not payload["text"].strip():
        print("[WARN] 没有输入文本，请使用 --text、--input-file 或 --case-file。")
        return 0
    result = run_b0(
        payload["text"],
        case_id=payload["case_id"],
        ground_truth=payload["ground_truth"],
        mock=args.mock,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
