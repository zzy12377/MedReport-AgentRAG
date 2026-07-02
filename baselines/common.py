# -*- coding: utf-8 -*-
"""Shared helpers for baseline scripts."""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set


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


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
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
                yield obj


def completed_case_ids(path: str) -> Set[str]:
    ids: Set[str] = set()
    for row in read_jsonl(path) or []:
        case_id = str(row.get("case_id", "")).strip()
        status = str(row.get("status", "")).upper()
        if case_id and status != "FAILED":
            ids.add(case_id)
    return ids


def load_case_file(case_file: str) -> Dict[str, str]:
    with open(case_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    case_id = str(data.get("Participant No.") or data.get("id") or os.path.splitext(os.path.basename(case_file))[0])
    text = str(data.get("Text") or data.get("Symptoms") or data)
    truth = str(data.get("Diagnosis") or data.get("Processed Diagnosis") or "")
    return {"case_id": case_id, "text": text, "ground_truth": truth}


def natural_sort_key(path: str) -> List[Any]:
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def iter_case_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        print(f"[WARN] 测试集目录不存在：{folder}")
        print("请先运行：python scripts/prepare_ddxplus_for_medrag.py")
        return []
    return sorted(
        [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.endswith(".json") and os.path.isfile(os.path.join(folder, name))
        ],
        key=natural_sort_key,
    )


def parse_limit(limit: str | int | None, total: int) -> int:
    if limit is None:
        return total
    if isinstance(limit, int):
        return total if limit <= 0 else min(limit, total)
    text = str(limit).strip().lower()
    if text in {"all", "0", "-1", "none"}:
        return total
    try:
        value = int(text)
    except ValueError:
        return total
    return total if value <= 0 else min(value, total)


def load_case_payloads(test_dir: str, limit: str | int | None = "all") -> List[Dict[str, str]]:
    files = iter_case_files(test_dir)
    max_count = parse_limit(limit, len(files))
    payloads = []
    for path in files[:max_count]:
        try:
            payloads.append(load_case_file(path))
        except Exception as exc:
            payloads.append(
                {
                    "case_id": os.path.splitext(os.path.basename(path))[0],
                    "text": "",
                    "ground_truth": "",
                    "error": str(exc),
                }
            )
    return payloads


def read_text_input(text: Optional[str] = None, input_file: Optional[str] = None, case_file: Optional[str] = None) -> Dict[str, str]:
    if case_file:
        return load_case_file(case_file)
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            return {"case_id": os.path.splitext(os.path.basename(input_file))[0], "text": f.read(), "ground_truth": ""}
    return {"case_id": "manual", "text": text or "", "ground_truth": ""}
