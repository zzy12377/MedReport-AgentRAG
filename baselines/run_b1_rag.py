# -*- coding: utf-8 -*-
"""B1 baseline: FAISS similar-case retrieval plus LLM."""

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
from engines.ner.medical_ner import entities_to_query_text, extract_medical_entities
from engines.retrieval.faiss_retriever import DATA_PREP_HINT, FaissCaseRetriever


SYSTEM_PROMPT = (
    "You are a cautious RAG-based medical decision-support assistant. "
    "Use the retrieved similar cases as references, but do not invent patient facts."
)


def _format_retrieved_cases(retrieved_cases: list) -> str:
    if not retrieved_cases:
        return "None"
    blocks = []
    for idx, case in enumerate(retrieved_cases, start=1):
        raw = str(case.get("raw_text", ""))[:1200]
        blocks.append(
            f"[Case {idx}] id={case.get('case_id')} diagnosis={case.get('diagnosis')} "
            f"similarity={case.get('similarity'):.4f}\n{raw}"
        )
    return "\n\n".join(blocks)


def run_b1(
    case_text: str,
    case_id: str = "manual",
    ground_truth: str = "",
    top_k: int = 3,
    mock: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    entities = extract_medical_entities(case_text)
    query_text = entities_to_query_text(entities) or case_text

    retrieved_cases = []
    try:
        retriever = FaissCaseRetriever(top_k=top_k, force_local=True)
        retrieved_cases = retriever.retrieve_similar_cases(query_text, top_k=top_k)
    except Exception as exc:
        print(f"[WARN] B1 检索不可用：{exc}")
        print(DATA_PREP_HINT)

    prompt = (
        f"Patient text:\n{case_text}\n\n"
        f"Extracted entities:\n{json.dumps(entities, ensure_ascii=False)}\n\n"
        f"Retrieved similar cases:\n{_format_retrieved_cases(retrieved_cases)}\n\n"
        "Please provide the most likely diagnosis and brief reasoning."
    )
    response = LLMGateway(mock=mock).generate(prompt, system_prompt=SYSTEM_PROMPT, mode="B1")
    result = make_standard_result(
        case_id=case_id,
        mode="B1",
        prediction=extract_prediction(response),
        ground_truth=ground_truth,
        retrieved_cases=retrieved_cases,
    )
    result["entities"] = entities
    result["llm_response"] = response
    if output_dir:
        result["output_files"] = save_result(result, output_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B1 RAG baseline.")
    parser.add_argument("--text", default="", help="Manual case text.")
    parser.add_argument("--input-file", default=None, help="Plain text input file.")
    parser.add_argument("--case-file", default=None, help="DDXPlus participant JSON file.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--mock", action="store_true", help="Force mock LLM output.")
    parser.add_argument("--output-dir", default="./storage/results", help="Where JSON/CSV outputs are saved.")
    args = parser.parse_args()

    payload = read_text_input(args.text, args.input_file, args.case_file)
    if not payload["text"].strip():
        print("[WARN] 没有输入文本，请使用 --text、--input-file 或 --case-file。")
        return 0
    result = run_b1(
        payload["text"],
        case_id=payload["case_id"],
        ground_truth=payload["ground_truth"],
        top_k=args.top_k,
        mock=args.mock,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
