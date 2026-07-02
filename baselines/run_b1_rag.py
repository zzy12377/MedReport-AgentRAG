# -*- coding: utf-8 -*-
"""B1 baseline: FAISS similar-case retrieval plus LLM.

Single-case mode remains compatible with phase 1. Batch mode is enabled when no
manual input is provided and writes one JSON object per test case to JSONL.
"""

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

from baselines.common import (
    SAFETY_NOTE,
    append_jsonl,
    completed_case_ids,
    load_case_payloads,
    make_standard_result,
    read_text_input,
    save_result,
)
from engines.llm.llm_gateway import LLMGateway, extract_prediction
from engines.ner.medical_ner import entities_to_query_text, extract_medical_entities
from engines.retrieval.faiss_retriever import DATA_PREP_HINT, FaissCaseRetriever
from engines.retrieval.multi_source_retriever import MultiSourceRetriever, normalize_vector_sources


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
        similarity = case.get("similarity", 0.0)
        blocks.append(
            f"[Case {idx}] id={case.get('case_id')} diagnosis={case.get('diagnosis')} "
            f"similarity={float(similarity):.4f}\n{raw}"
        )
    return "\n\n".join(blocks)


def run_b1(
    case_text: str,
    case_id: str = "manual",
    ground_truth: str = "",
    top_k: int = 3,
    mock: bool = False,
    output_dir: Optional[str] = None,
    retriever: Optional[Any] = None,
    vector_sources: Optional[list[str]] = None,
    top_k_per_source: Optional[int] = None,
    mode: str = "B1",
) -> Dict[str, Any]:
    entities = extract_medical_entities(case_text)
    query_text = entities_to_query_text(entities) or case_text

    retrieved_cases = []
    try:
        active_retriever = retriever or FaissCaseRetriever(top_k=top_k, force_local=True)
        if isinstance(active_retriever, MultiSourceRetriever):
            retrieved_cases = active_retriever.retrieve(
                query_text,
                sources=vector_sources,
                top_k=top_k,
                top_k_per_source=top_k_per_source,
            )
        else:
            retrieved_cases = active_retriever.retrieve_similar_cases(query_text, top_k=top_k)
    except Exception as exc:
        print(f"[WARN] B1 检索不可用：{exc}")
        print(DATA_PREP_HINT)

    prompt = (
        f"Patient text:\n{case_text}\n\n"
        f"Extracted entities:\n{json.dumps(entities, ensure_ascii=False)}\n\n"
        f"Retrieved similar cases:\n{_format_retrieved_cases(retrieved_cases)}\n\n"
        "Please provide the most likely diagnosis and brief reasoning."
    )
    response = LLMGateway(mock=mock).generate(prompt, system_prompt=SYSTEM_PROMPT, mode=mode)
    result = make_standard_result(
        case_id=case_id,
        mode=mode,
        prediction=extract_prediction(response),
        ground_truth=ground_truth,
        retrieved_cases=retrieved_cases,
    )
    result["entities"] = entities
    result["llm_response"] = response
    result["retrieval_mode"] = "multi_vector" if isinstance(retriever, MultiSourceRetriever) else "ddxplus_faiss"
    result["retrieval_sources"] = normalize_vector_sources(vector_sources) or ["all"] if isinstance(retriever, MultiSourceRetriever) else ["ddxplus_cases"]
    result["status"] = "SUCCESS"
    if output_dir:
        result["output_files"] = save_result(result, output_dir)
    return result


def _failed_row(case_id: str, error: str, mode: str = "B1_RAG") -> Dict[str, Any]:
    return {
        "case_id": str(case_id),
        "mode": mode,
        "error": str(error),
        "status": "FAILED",
        "prediction": "",
        "ground_truth": "",
        "retrieved_cases": [],
        "kg_evidence": [],
        "agent_outputs": [],
        "critique": {},
        "safety_note": SAFETY_NOTE,
    }


def run_b1_batch(
    test_dir: str = "./dataset/df/test",
    output: str = "./storage/results/b1_rag_results.jsonl",
    limit: str = "all",
    top_k: int = 5,
    resume: bool = False,
    mock: bool = False,
    retries: int = 1,
    vector_sources: Optional[list[str]] = None,
    vector_base_dir: str = "./vector_db",
    top_k_per_source: Optional[int] = None,
) -> Dict[str, Any]:
    payloads = load_case_payloads(test_dir, limit=limit)
    if not payloads:
        return {"total": 0, "success": 0, "failed": 0, "output": output}

    done = completed_case_ids(output) if resume else set()
    retriever: Any
    if vector_sources:
        retriever = MultiSourceRetriever(base_dir=vector_base_dir, force_local=True)
        print(f"[INFO] B1 multi-vector sources: {normalize_vector_sources(vector_sources) or 'all'}")
    else:
        retriever = FaissCaseRetriever(top_k=top_k, force_local=True)
    success = 0
    failed = 0
    skipped = 0

    for idx, payload in enumerate(payloads, start=1):
        case_id = payload.get("case_id", f"case_{idx}")
        if resume and case_id in done:
            skipped += 1
            print(f"[SKIP] {idx}/{len(payloads)} case_id={case_id} already completed")
            continue
        print(f"[B1] {idx}/{len(payloads)} case_id={case_id}")

        last_error = ""
        for attempt in range(1, max(1, retries) + 1):
            try:
                if payload.get("error"):
                    raise RuntimeError(payload["error"])
                if not str(payload.get("text", "")).strip():
                    raise RuntimeError("empty case text")
                row = run_b1(
                    payload["text"],
                    case_id=case_id,
                    ground_truth=payload.get("ground_truth", ""),
                    top_k=top_k,
                    mock=mock,
                    output_dir=None,
                    retriever=retriever,
                    vector_sources=vector_sources,
                    top_k_per_source=top_k_per_source,
                    mode="B1_RAG",
                )
                append_jsonl(output, row)
                success += 1
                break
            except Exception as exc:
                last_error = str(exc)
                print(f"[WARN] case_id={case_id} attempt {attempt}/{max(1, retries)} failed: {last_error}")
        else:
            append_jsonl(output, _failed_row(case_id, last_error, mode="B1_RAG"))
            failed += 1

    summary = {"total": len(payloads), "success": success, "failed": failed, "skipped": skipped, "output": output}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B1 RAG baseline.")
    parser.add_argument("--text", default="", help="Manual case text. If omitted, batch mode runs over test_dir.")
    parser.add_argument("--input-file", default=None, help="Plain text input file.")
    parser.add_argument("--case-file", default=None, help="DDXPlus participant JSON file.")
    parser.add_argument("--test-dir", default="./dataset/df/test", help="DDXPlus test JSON directory for batch mode.")
    parser.add_argument("--limit", default="all", help="'all' or an integer count for batch mode.")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Skip completed case_id values in output JSONL.")
    parser.add_argument("--retries", type=int, default=1, help="Per-case retry count in batch mode.")
    parser.add_argument("--mock", action="store_true", help="Force mock LLM output.")
    parser.add_argument(
        "--vector-sources",
        nargs="+",
        default=None,
        help="Optional vector_db sources for multi-source retrieval, e.g. ddxplus_cases ddxplus_kg or all.",
    )
    parser.add_argument("--vector-base-dir", default="./vector_db", help="Base directory for multi-source vector stores.")
    parser.add_argument("--top-k-per-source", type=int, default=None, help="Per-source top-k for multi-source retrieval.")
    parser.add_argument("--output", default="./storage/results/b1_rag_results.jsonl", help="Batch JSONL output.")
    parser.add_argument("--output-dir", default="./storage/results", help="Single-case JSON/CSV output directory.")
    args = parser.parse_args()

    single_mode = bool(args.text or args.input_file or args.case_file)
    if not single_mode:
        run_b1_batch(
            test_dir=args.test_dir,
            output=args.output,
            limit=args.limit,
            top_k=args.top_k,
            resume=args.resume,
            mock=args.mock,
            retries=args.retries,
            vector_sources=args.vector_sources,
            vector_base_dir=args.vector_base_dir,
            top_k_per_source=args.top_k_per_source,
        )
        return 0

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
        retriever=MultiSourceRetriever(base_dir=args.vector_base_dir, force_local=True) if args.vector_sources else None,
        vector_sources=args.vector_sources,
        top_k_per_source=args.top_k_per_source,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
