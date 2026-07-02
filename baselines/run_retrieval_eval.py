# -*- coding: utf-8 -*-
"""Non-LLM retrieval evaluation over the DDXPlus test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common import load_case_payloads
from engines.retrieval.faiss_retriever import FaissCaseRetriever


def _norm(value: Any) -> str:
    return str(value or "").replace("_", " ").replace("-", " ").strip().lower()


def _hit_at(retrieved: List[Dict[str, Any]], truth: str, k: int) -> bool:
    truth_n = _norm(truth)
    if not truth_n:
        return False
    for row in retrieved[:k]:
        diagnosis = _norm(row.get("diagnosis"))
        if diagnosis and (truth_n == diagnosis or truth_n in diagnosis or diagnosis in truth_n):
            return True
    return False


def run_retrieval_eval(
    test_dir: str = "./dataset/df/test",
    limit: str = "all",
    top_k: int = 5,
    output: str = "./storage/results/retrieval_eval.json",
    details_output: str = "./storage/results/retrieval_eval_details.jsonl",
) -> Dict[str, Any]:
    payloads = [p for p in load_case_payloads(test_dir, limit=limit) if p.get("text")]
    if not payloads:
        summary = {"mode": "RETRIEVAL_EVAL", "total_cases": 0, "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    retriever = FaissCaseRetriever(top_k=top_k, force_local=True)
    queries = [payload["text"] for payload in payloads]
    print(f"[INFO] Embedding {len(queries)} test queries for retrieval eval...")
    query_vectors = retriever.embedding_engine.embed_texts(queries)
    all_results = retriever.search_vectors(query_vectors, top_k=max(5, top_k))

    os.makedirs(os.path.dirname(details_output) or ".", exist_ok=True)
    hits = {1: 0, 3: 0, 5: 0}
    with open(details_output, "w", encoding="utf-8") as f:
        for idx, (payload, retrieved) in enumerate(zip(payloads, all_results), start=1):
            truth = payload.get("ground_truth", "")
            for k in hits:
                hits[k] += int(_hit_at(retrieved, truth, k))
            row = {
                "mode": "RETRIEVAL_EVAL",
                "case_id": payload.get("case_id"),
                "ground_truth": truth,
                "retrieved_cases": retrieved[: max(5, top_k)],
                "hit@1": _hit_at(retrieved, truth, 1),
                "hit@3": _hit_at(retrieved, truth, 3),
                "hit@5": _hit_at(retrieved, truth, 5),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if idx % 25 == 0 or idx == len(payloads):
                print(f"[EVAL] {idx}/{len(payloads)}")

    total = len(payloads)
    summary = {
        "mode": "RETRIEVAL_EVAL",
        "total_cases": total,
        "Recall@1": round(hits[1] / total, 4),
        "Recall@3": round(hits[3] / total, 4),
        "Recall@5": round(hits[5] / total, 4),
        "details_output": details_output,
    }
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[INFO] Retrieval eval saved to: {output}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run non-LLM retrieval evaluation.")
    parser.add_argument("--test-dir", default="./dataset/df/test")
    parser.add_argument("--limit", default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="./storage/results/retrieval_eval.json")
    parser.add_argument("--details-output", default="./storage/results/retrieval_eval_details.jsonl")
    args = parser.parse_args()
    run_retrieval_eval(
        test_dir=args.test_dir,
        limit=args.limit,
        top_k=args.top_k,
        output=args.output,
        details_output=args.details_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
