# -*- coding: utf-8 -*-
"""Shared helpers for baseline scripts."""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


SAFETY_NOTE = "本结果仅用于课程演示和辅助参考，不能替代医生诊断。"


def make_standard_result(
    case_id: str,
    mode: str,
    prediction: str,
    ground_truth: str = "",
    retrieved_cases: Optional[List[Dict[str, Any]]] = None,
    kg_evidence: Optional[List[Dict[str, Any]]] = None,
    agent_outputs: Optional[List[Dict[str, Any]]] = None,
    critique: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "case_id": str(case_id),
        "mode": mode,
        "prediction": prediction or "",
        "ground_truth": ground_truth or "",
        "l1_pred": "",
        "l2_pred": "",
        "l3_pred": prediction or "",
        "l1_truth": "",
        "l2_truth": "",
        "l3_truth": ground_truth or "",
        "retrieved_cases": retrieved_cases or [],
        "kg_evidence": kg_evidence or [],
        "agent_outputs": agent_outputs or [],
        "critique": critique or {},
        "safety_note": SAFETY_NOTE,
    }


def save_result(result: Dict[str, Any], output_dir: str = "./storage/results") -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{result.get('mode', 'baseline')}_{result.get('case_id', 'case')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    json_path = os.path.join(output_dir, safe_stem + ".json")
    csv_path = os.path.join(output_dir, safe_stem + ".csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    fieldnames = list(result.keys())
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row = {
            key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            for key, value in result.items()
        }
        writer.writerow(row)

    return {"json": json_path, "csv": csv_path}


def load_case_file(case_file: str) -> Dict[str, str]:
    with open(case_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    case_id = str(data.get("Participant No.") or data.get("id") or os.path.splitext(os.path.basename(case_file))[0])
    text = str(data.get("Text") or data.get("Symptoms") or data)
    truth = str(data.get("Diagnosis") or data.get("Processed Diagnosis") or "")
    return {"case_id": case_id, "text": text, "ground_truth": truth}


def read_text_input(text: Optional[str] = None, input_file: Optional[str] = None, case_file: Optional[str] = None) -> Dict[str, str]:
    if case_file:
        return load_case_file(case_file)
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            return {"case_id": os.path.splitext(os.path.basename(input_file))[0], "text": f.read(), "ground_truth": ""}
    return {"case_id": "manual", "text": text or "", "ground_truth": ""}

