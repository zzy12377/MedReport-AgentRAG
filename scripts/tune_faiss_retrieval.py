# -*- coding: utf-8 -*-
"""Run lightweight top-k tuning experiments for the multi-vector retriever.

This script does not call an LLM. It records how different top_k and
top_k_per_source settings change score distribution and top results, giving
the RAG module a small reproducible tuning table for reports or defense.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engines.retrieval.multi_source_retriever import MultiSourceRetriever


DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "storage", "metrics", "faiss_tuning_results.csv")
DEFAULT_QUERIES = [
    {
        "query_id": "q_tuberculosis",
        "query": "18-year-old male with fever cough sore throat and night sweats",
        "expected": "Tuberculosis",
    },
    {
        "query_id": "q_angina",
        "query": "chest pain on exertion relieved by rest with hypertension and high cholesterol",
        "expected": "angina",
    },
    {
        "query_id": "q_metabolic",
        "query": "ALT 85.2 U/L GLU 7.2 mmol/L LDL-C 4.1 mmol/L blood pressure 150/95 mmHg",
        "expected": "",
    },
]


def _parse_int_list(values: Iterable[str]) -> List[int]:
    parsed: List[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    return parsed


def _load_queries(path: Optional[str], inline_queries: Optional[List[str]]) -> List[Dict[str, str]]:
    if inline_queries:
        return [
            {"query_id": f"query_{idx}", "query": query, "expected": ""}
            for idx, query in enumerate(inline_queries, start=1)
        ]
    if not path:
        return list(DEFAULT_QUERIES)
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                rows.append(
                    {
                        "query_id": str(obj.get("query_id") or obj.get("id") or f"query_{idx}"),
                        "query": str(obj.get("query") or obj.get("text") or ""),
                        "expected": str(obj.get("expected") or obj.get("diagnosis") or ""),
                    }
                )
            else:
                rows.append({"query_id": f"query_{idx}", "query": line, "expected": ""})
    return [row for row in rows if row.get("query")]


def _hit_expected(results: List[Dict[str, Any]], expected: str) -> bool:
    expected = str(expected or "").strip().lower()
    if not expected:
        return False
    for row in results:
        haystack = " ".join(
            [
                str(row.get("diagnosis") or ""),
                str(row.get("title") or ""),
                str(row.get("raw_text") or ""),
            ]
        ).lower()
        if expected in haystack:
            return True
    return False


def _source_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values or any(str(value).lower() == "all" for value in values):
        return None
    return values


def run_tuning(
    queries: List[Dict[str, str]],
    sources: Optional[List[str]],
    top_k_values: List[int],
    top_k_per_source_values: List[int],
    output: str,
    vector_db: str,
    local: bool,
) -> List[Dict[str, Any]]:
    retriever = MultiSourceRetriever(base_dir=vector_db, force_local=local)
    if not retriever.sources:
        print(f"[WARN] No vector stores found under {vector_db}.")
        print("Next step: python scripts/build_vector_stores.py --local")
        return []

    rows: List[Dict[str, Any]] = []
    active_sources = _source_list(sources)
    print(f"[INFO] Sources: {active_sources or 'all'}")
    print(f"[INFO] Available stores: {', '.join(retriever.sources)}")

    for query_row in queries:
        query_id = query_row["query_id"]
        query = query_row["query"]
        expected = query_row.get("expected", "")
        print(f"[INFO] Query {query_id}: {query}")
        for top_k in top_k_values:
            for top_k_per_source in top_k_per_source_values:
                try:
                    results = retriever.retrieve(
                        query_text=query,
                        sources=active_sources,
                        top_k=top_k,
                        top_k_per_source=top_k_per_source,
                    )
                    scores = [float(row.get("similarity", 0.0)) for row in results]
                    top1 = results[0] if results else {}
                    rows.append(
                        {
                            "query_id": query_id,
                            "query": query,
                            "expected": expected,
                            "sources": "all" if active_sources is None else ";".join(active_sources),
                            "top_k": top_k,
                            "top_k_per_source": top_k_per_source,
                            "result_count": len(results),
                            "top1_score": round(scores[0], 6) if scores else 0.0,
                            "avg_score": round(sum(scores) / len(scores), 6) if scores else 0.0,
                            "min_score": round(min(scores), 6) if scores else 0.0,
                            "top1_source": top1.get("source", ""),
                            "top1_diagnosis": top1.get("diagnosis", ""),
                            "top1_title": top1.get("title", ""),
                            "expected_in_topk": _hit_expected(results, expected),
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "query_id": query_id,
                            "query": query,
                            "expected": expected,
                            "sources": "all" if active_sources is None else ";".join(active_sources),
                            "top_k": top_k,
                            "top_k_per_source": top_k_per_source,
                            "result_count": 0,
                            "top1_score": 0.0,
                            "avg_score": 0.0,
                            "min_score": 0.0,
                            "top1_source": "",
                            "top1_diagnosis": "",
                            "top1_title": "",
                            "expected_in_topk": False,
                            "error": str(exc),
                        }
                    )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fieldnames = [
        "query_id",
        "query",
        "expected",
        "sources",
        "top_k",
        "top_k_per_source",
        "result_count",
        "top1_score",
        "avg_score",
        "min_score",
        "top1_source",
        "top1_diagnosis",
        "top1_title",
        "expected_in_topk",
        "error",
    ]
    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"[INFO] Tuning table saved to: {output}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune top-k settings for multi-vector FAISS retrieval.")
    parser.add_argument("--queries-file", default=None, help="JSONL or plain-text query file.")
    parser.add_argument("--query", action="append", help="Inline query. Can be passed multiple times.")
    parser.add_argument("--sources", nargs="+", default=["all"], help="Vector sources, or all.")
    parser.add_argument("--top-k-values", nargs="+", default=["3", "5", "10"], help="Values like 3 5 10 or 3,5,10.")
    parser.add_argument(
        "--top-k-per-source-values",
        nargs="+",
        default=["2", "3", "5"],
        help="Values like 2 3 5 or 2,3,5.",
    )
    parser.add_argument("--vector-db", default=os.path.join(PROJECT_ROOT, "vector_db"), help="Vector DB directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV output path.")
    parser.add_argument("--local", action="store_true", help="Force local embedding.")
    args = parser.parse_args()

    queries = _load_queries(args.queries_file, args.query)
    if not queries:
        print("[WARN] No queries provided.")
        return 0
    run_tuning(
        queries=queries,
        sources=args.sources,
        top_k_values=_parse_int_list(args.top_k_values),
        top_k_per_source_values=_parse_int_list(args.top_k_per_source_values),
        output=args.output,
        vector_db=args.vector_db,
        local=args.local,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
