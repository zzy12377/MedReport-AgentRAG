# -*- coding: utf-8 -*-
"""Evaluate rule-based medical NER with Precision/Recall/F1.

The default gold file is a small manually labeled demonstration set under
data/processed/. It is intentionally lightweight so it can run without OCR,
LLM, FAISS, Redis, Gradio, or external network access.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from engines.ner.medical_ner import extract_medical_entities


DEFAULT_GOLD_PATH = os.path.join(PROJECT_ROOT, "tests", "fixtures", "ner_eval_samples.jsonl")
DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "storage", "metrics", "ner_eval_summary.json")
DEFAULT_DETAILS_PATH = os.path.join(PROJECT_ROOT, "storage", "metrics", "ner_eval_details.csv")


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _norm_unit(unit: Any) -> str:
    text = str(unit or "").strip().replace("μ", "u").replace("µ", "u")
    lower = text.replace(" ", "").lower()
    mapping = {
        "u/l": "U/L",
        "iu/l": "IU/L",
        "mmol/l": "mmol/L",
        "umol/l": "umol/L",
        "pmol/l": "pmol/L",
        "miu/l": "mIU/L",
        "g/l": "g/L",
        "g/dl": "g/dL",
        "mg/dl": "mg/dL",
        "mmhg": "mmHg",
        "kg/m2": "kg/m2",
        "kg/m²": "kg/m2",
        "bpm": "bpm",
        "次/分": "bpm",
        "x10^9/l": "10^9/L",
        "10^9/l": "10^9/L",
        "x10^12/l": "10^12/L",
        "10^12/l": "10^12/L",
        "%": "%",
    }
    return mapping.get(lower, text)


def _same_value(expected: Dict[str, Any], predicted: Dict[str, Any], tolerance: float) -> bool:
    if expected.get("value") is None:
        return True
    try:
        return abs(float(expected.get("value")) - float(predicted.get("value"))) <= tolerance
    except Exception:
        return False


def _match_entities(
    expected: Iterable[Dict[str, Any]],
    predicted: Iterable[Dict[str, Any]],
    value_tolerance: float,
    check_unit: bool,
) -> Tuple[int, int, int, List[Dict[str, Any]], List[Dict[str, Any]]]:
    expected_rows = [dict(row) for row in expected]
    predicted_rows = [dict(row) for row in predicted]
    matched_predicted: set[int] = set()
    matched_expected: set[int] = set()

    for e_idx, exp in enumerate(expected_rows):
        for p_idx, pred in enumerate(predicted_rows):
            if p_idx in matched_predicted:
                continue
            if str(exp.get("name")) != str(pred.get("name")):
                continue
            if not _same_value(exp, pred, value_tolerance):
                continue
            if check_unit and _norm_unit(exp.get("unit")) != _norm_unit(pred.get("unit")):
                continue
            matched_expected.add(e_idx)
            matched_predicted.add(p_idx)
            break

    tp = len(matched_expected)
    fp = len(predicted_rows) - len(matched_predicted)
    fn = len(expected_rows) - len(matched_expected)
    missing = [row for idx, row in enumerate(expected_rows) if idx not in matched_expected]
    extra = [row for idx, row in enumerate(predicted_rows) if idx not in matched_predicted]
    return tp, fp, fn, missing, extra


def _safe_div(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def evaluate(gold_path: str, value_tolerance: float = 0.01, check_unit: bool = False) -> Dict[str, Any]:
    rows = _load_jsonl(gold_path)
    totals = {"tp": 0, "fp": 0, "fn": 0}
    details: List[Dict[str, Any]] = []

    for row in rows:
        case_id = str(row.get("case_id") or f"case_{len(details) + 1}")
        text = str(row.get("text") or "")
        expected = list(row.get("expected") or [])
        predicted = extract_medical_entities(text)
        tp, fp, fn, missing, extra = _match_entities(expected, predicted, value_tolerance, check_unit)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        details.append(
            {
                "case_id": case_id,
                "expected_count": len(expected),
                "predicted_count": len(predicted),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "missing": json.dumps(missing, ensure_ascii=False),
                "extra": json.dumps(
                    [
                        {
                            "name": item.get("name"),
                            "value": item.get("value"),
                            "unit": item.get("unit"),
                            "original_text": item.get("original_text"),
                        }
                        for item in extra
                    ],
                    ensure_ascii=False,
                ),
            }
        )

    precision = _safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = _safe_div(totals["tp"], totals["tp"] + totals["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "gold_path": gold_path,
        "total_cases": len(rows),
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "details": details,
    }


def _write_outputs(summary: Dict[str, Any], output_path: str, details_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(details_path) or ".", exist_ok=True)
    details = list(summary.get("details") or [])
    summary_for_json = dict(summary)
    summary_for_json["details_csv"] = details_path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_for_json, f, ensure_ascii=False, indent=2)
    fieldnames = [
        "case_id",
        "expected_count",
        "predicted_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "missing",
        "extra",
    ]
    with open(details_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate medical_ner.py with Precision/Recall/F1.")
    parser.add_argument("--gold", default=DEFAULT_GOLD_PATH, help="Gold JSONL with text and expected entities.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Summary JSON output path.")
    parser.add_argument("--details-csv", default=DEFAULT_DETAILS_PATH, help="Per-case CSV output path.")
    parser.add_argument("--value-tolerance", type=float, default=0.01, help="Allowed numeric value difference.")
    parser.add_argument("--check-unit", action="store_true", help="Require unit equality in addition to name/value.")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        print(f"[WARN] NER gold file not found: {args.gold}")
        print("Next step: create data/processed/ner_eval_samples.jsonl with text and expected entity lists.")
        return 0

    summary = evaluate(args.gold, value_tolerance=args.value_tolerance, check_unit=args.check_unit)
    _write_outputs(summary, args.output, args.details_csv)
    printable = {k: v for k, v in summary.items() if k != "details"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    print(f"[INFO] Summary saved to: {args.output}")
    print(f"[INFO] Details saved to: {args.details_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
