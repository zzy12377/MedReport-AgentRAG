# -*- coding: utf-8 -*-
"""B2 baseline: FAISS + KG evidence + agent skeleton with JSONL resume."""

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
from engines.agents.agent_pipeline import run_agent_pipeline
from engines.kg.kg_extractor import DDXPlusKGRetriever
from engines.llm.llm_gateway import LLMGateway, extract_prediction
from engines.ner.medical_ner import entities_to_query_text, extract_medical_entities
from engines.retrieval.faiss_retriever import FaissCaseRetriever
from engines.retrieval.multi_source_retriever import MultiSourceRetriever, normalize_vector_sources


SYSTEM_PROMPT = (
    "You are a cautious KG-RAG medical decision-support assistant. "
    "Use similar cases, KG evidence and specialist-agent outputs."
)


def run_b2(
    case_text: str,
    case_id: str = "manual",
    ground_truth: str = "",
    top_k: int = 5,
    kg_top_k: int = 10,
    mock: bool = True,
    output_dir: Optional[str] = None,
    retriever: Optional[Any] = None,
    kg_retriever: Optional[DDXPlusKGRetriever] = None,
    vector_sources: Optional[list[str]] = None,
    top_k_per_source: Optional[int] = None,
    mode: str = "B2",
) -> Dict[str, Any]:
    entities = extract_medical_entities(case_text)
    query_text = entities_to_query_text(entities) or case_text
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

    active_kg = kg_retriever or DDXPlusKGRetriever()
    kg_evidence = active_kg.retrieve(query_text=case_text, entities=entities, top_k=kg_top_k)

    agent_summary = run_agent_pipeline(entities, retrieved_cases)
    agent_outputs = agent_summary.get("agent_outputs", [])
    critique = agent_summary.get("critique", {})

    prompt = (
        f"Patient text:\n{case_text}\n\n"
        f"Entities:\n{json.dumps(entities, ensure_ascii=False)}\n\n"
        f"Retrieved cases:\n{json.dumps(retrieved_cases[:top_k], ensure_ascii=False)}\n\n"
        f"KG evidence:\n{json.dumps(kg_evidence, ensure_ascii=False)}\n\n"
        f"Agent outputs:\n{json.dumps(agent_outputs, ensure_ascii=False)}\n\n"
        f"Critique:\n{json.dumps(critique, ensure_ascii=False)}\n\n"
        "Please provide a concise final diagnosis."
    )
    response = LLMGateway(mock=mock).generate(prompt, system_prompt=SYSTEM_PROMPT, mode=mode)
    result = make_standard_result(
        case_id=case_id,
        mode=mode,
        prediction=extract_prediction(response),
        ground_truth=ground_truth,
        retrieved_cases=retrieved_cases,
        kg_evidence=kg_evidence,
        agent_outputs=agent_outputs,
        critique=critique,
    )
    result["entities"] = entities
    result["llm_response"] = response
    result["retrieval_mode"] = "multi_vector" if isinstance(retriever, MultiSourceRetriever) else "ddxplus_faiss"
    result["retrieval_sources"] = normalize_vector_sources(vector_sources) or ["all"] if isinstance(retriever, MultiSourceRetriever) else ["ddxplus_cases"]
    result["status"] = "SUCCESS"
    if output_dir:
        result["output_files"] = save_result(result, output_dir)
    return result


def _failed_row(case_id: str, error: str, mode: str = "B2_KG_RAG") -> Dict[str, Any]:
    return {
        "case_id": str(case_id),
        "mode": mode,
        "prediction": "",
        "ground_truth": "",
        "retrieved_cases": [],
        "kg_evidence": [],
        "agent_outputs": [],
        "critique": {},
        "safety_note": SAFETY_NOTE,
        "status": "FAILED",
        "error": str(error),
    }


def run_b2_batch(
    test_dir: str = "./dataset/df/test",
    output: str = "./storage/results/b2_kg_rag_results.jsonl",
    limit: str = "all",
    top_k: int = 5,
    kg_top_k: int = 10,
    resume: bool = False,
    mock: bool = True,
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
        print(f"[INFO] B2 multi-vector sources: {normalize_vector_sources(vector_sources) or 'all'}")
    else:
        retriever = FaissCaseRetriever(top_k=top_k, force_local=True)
    kg_retriever = DDXPlusKGRetriever()
    success = 0
    failed = 0
    skipped = 0

    for idx, payload in enumerate(payloads, start=1):
        case_id = payload.get("case_id", f"case_{idx}")
        if resume and case_id in done:
            skipped += 1
            print(f"[SKIP] {idx}/{len(payloads)} case_id={case_id} already completed")
            continue
        print(f"[B2] {idx}/{len(payloads)} case_id={case_id}")

        last_error = ""
        for attempt in range(1, max(1, retries) + 1):
            try:
                if payload.get("error"):
                    raise RuntimeError(payload["error"])
                if not str(payload.get("text", "")).strip():
                    raise RuntimeError("empty case text")
                row = run_b2(
                    payload["text"],
                    case_id=case_id,
                    ground_truth=payload.get("ground_truth", ""),
                    top_k=top_k,
                    kg_top_k=kg_top_k,
                    mock=mock,
                    output_dir=None,
                    retriever=retriever,
                    kg_retriever=kg_retriever,
                    vector_sources=vector_sources,
                    top_k_per_source=top_k_per_source,
                    mode="B2_KG_RAG",
                )
                append_jsonl(output, row)
                success += 1
                break
            except Exception as exc:
                last_error = str(exc)
                print(f"[WARN] case_id={case_id} attempt {attempt}/{max(1, retries)} failed: {last_error}")
        else:
            append_jsonl(output, _failed_row(case_id, last_error, mode="B2_KG_RAG"))
            failed += 1

    summary = {"total": len(payloads), "success": success, "failed": failed, "skipped": skipped, "output": output}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run B2 KG-RAG baseline.")
    parser.add_argument("--text", default="")
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--case-file", default=None)
    parser.add_argument("--test-dir", default="./dataset/df/test")
    parser.add_argument("--limit", default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--kg-top-k", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument(
        "--vector-sources",
        nargs="+",
        default=None,
        help="Optional vector_db sources for multi-source retrieval, e.g. ddxplus_cases ddxplus_kg or all.",
    )
    parser.add_argument("--vector-base-dir", default="./vector_db")
    parser.add_argument("--top-k-per-source", type=int, default=None)
    parser.add_argument("--output", default="./storage/results/b2_kg_rag_results.jsonl")
    parser.add_argument("--output-dir", default="./storage/results")
    args = parser.parse_args()

    single_mode = bool(args.text or args.input_file or args.case_file)
    if not single_mode:
        run_b2_batch(
            test_dir=args.test_dir,
            output=args.output,
            limit=args.limit,
            top_k=args.top_k,
            kg_top_k=args.kg_top_k,
            resume=args.resume,
            mock=args.mock,
            retries=args.retries,
            vector_sources=args.vector_sources,
            vector_base_dir=args.vector_base_dir,
            top_k_per_source=args.top_k_per_source,
        )
        return 0

    payload = read_text_input(args.text, args.input_file, args.case_file)
    result = run_b2(
        payload["text"],
        payload["case_id"],
        payload["ground_truth"],
        top_k=args.top_k,
        kg_top_k=args.kg_top_k,
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
