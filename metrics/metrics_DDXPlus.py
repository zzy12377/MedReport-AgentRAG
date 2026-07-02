# -*- coding: utf-8 -*-
"""DDXPlus metrics for JSON/JSONL/CSV baseline outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _norm(value: Any) -> str:
    return str(value or "").replace("_", " ").replace("-", " ").strip().lower()


def _parse_json_field(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _load_result(path: str) -> List[Dict[str, Any]]:
    lower = path.lower()
    if lower.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, list) else [obj]
    if lower.endswith(".csv"):
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            rows = []
            for row in csv.DictReader(f):
                rows.append({k: _parse_json_field(v) for k, v in row.items()})
            return rows
    return []


def load_results(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pattern in paths:
        matches = glob.glob(pattern)
        if not matches and os.path.exists(pattern):
            matches = [pattern]
        for path in matches:
            rows.extend(_load_result(path))
    return rows


def _exact_match(row: Dict[str, Any]) -> int:
    truth = _norm(row.get("ground_truth") or row.get("l3_truth"))
    pred = _norm(row.get("prediction") or row.get("l3_pred"))
    if not truth:
        return 0
    return int(truth == pred or truth in pred or pred in truth)


def _retrieved(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = _parse_json_field(row.get("retrieved_cases", []))
    return value if isinstance(value, list) else []


def _kg_evidence(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = _parse_json_field(row.get("kg_evidence", []))
    return value if isinstance(value, list) else []


def _recall_at(row: Dict[str, Any], k: int) -> int:
    truth = _norm(row.get("ground_truth") or row.get("l3_truth"))
    if not truth:
        return 0
    for case in _retrieved(row)[:k]:
        if not isinstance(case, dict):
            continue
        diagnosis = _norm(case.get("diagnosis"))
        if diagnosis and (truth == diagnosis or truth in diagnosis or diagnosis in truth):
            return 1
    return 0


def compute_metrics(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("mode") or "UNKNOWN")].append(row)

    table = []
    for mode, items in sorted(grouped.items()):
        total = len(items)
        failed_items = [r for r in items if str(r.get("status", "")).upper() == "FAILED" or r.get("error")]
        success_items = [r for r in items if r not in failed_items]
        success = len(success_items)
        failed = len(failed_items)
        exact = sum(_exact_match(r) for r in success_items)
        recall_1 = sum(_recall_at(r, 1) for r in success_items)
        recall_3 = sum(_recall_at(r, 3) for r in success_items)
        recall_5 = sum(_recall_at(r, 5) for r in success_items)
        kg_non_empty = sum(1 for r in success_items if _kg_evidence(r))
        retrieved_total = sum(len(_retrieved(r)) for r in success_items)
        denom = success or 1

        table.append(
            {
                "mode": mode,
                "total_cases": total,
                "success_cases": success,
                "failed_cases": failed,
                "accuracy_exact_match": round(exact / denom, 4),
                "Recall@1": round(recall_1 / denom, 4),
                "Recall@3": round(recall_3 / denom, 4),
                "Recall@5": round(recall_5 / denom, 4),
                "kg_evidence_non_empty_rate": round(kg_non_empty / denom, 4),
                "average_retrieved_cases": round(retrieved_total / denom, 4),
            }
        )
    return table


def save_table(table: List[Dict[str, Any]], output: str) -> None:
    if not table:
        return
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate B0/B1/B2 comparison metrics.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["./storage/results/*.json", "./storage/results/*.jsonl"],
        help="Result JSON/JSONL/CSV files or globs.",
    )
    parser.add_argument("--output", default="./storage/metrics/metrics_summary.csv")
    args = parser.parse_args()

    rows = load_results(args.inputs)
    if not rows:
        print("[WARN] 没有找到 baseline 结果。请先运行 B0/B1/B2 或 retrieval eval。")
        return 0
    table = compute_metrics(rows)
    save_table(table, args.output)
    print(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"[INFO] Metrics saved to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
