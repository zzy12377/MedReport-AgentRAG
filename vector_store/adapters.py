# -*- coding: utf-8 -*-
"""
vector_store/adapters.py

把 7 种数据源统一转换成标准化记录：
{
    "id": "...",
    "source": "...",
    "title": "...",
    "text": "...",
    "diagnosis": "...",
    "metadata": {...}
}

每个 adapter 都容错：文件不存在 / 列缺失时打印清晰的 [WARN] 并返回 []。
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Dict, List, Optional


# ============================================================
# 通用辅助
# ============================================================


def _safe_read_csv(path: str, label: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] {label}: 文件不存在 -> {path}")
        return []
    try:
        import pandas as pd
        return pd.read_csv(path).to_dict(orient="records")
    except Exception as e:
        print(f"[WARN] {label}: 读取 CSV 失败 -> {e}")
        return []


def _safe_read_json(path: str, label: str) -> Any:
    if not os.path.exists(path):
        print(f"[WARN] {label}: 文件不存在 -> {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] {label}: 读取 JSON 失败 -> {e}")
        return None


def _safe_read_excel(path: str, label: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"[WARN] {label}: 文件不存在 -> {path}")
        return []
    try:
        import pandas as pd
        return pd.read_excel(path).to_dict(orient="records")
    except Exception as e:
        print(f"[WARN] {label}: 读取 Excel 失败 -> {e}")
        return []


def _natural_sort_key(path: str) -> List[Any]:
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def _clean_text(s: Any, max_chars: int = 4096) -> str:
    s = str(s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > max_chars:
        s = s[:max_chars]
    return s or "empty"


# ============================================================
# 1. ddxplus_cases
# ============================================================


def adapt_ddxplus_cases(
    train_dir: str = "./dataset/df/train",
    test_dir: str = "./dataset/df/test",
    max_per_split: Optional[int] = None,
    max_rows: Optional[int] = None,
    include_test: bool = False,
) -> List[dict]:
    """
    读取 dataset/df/train/ 和 dataset/df/test/ 下的 participant_*.json。
    训练集病例保留 Diagnosis 作为历史病例标签（存入 metadata），测试集的 Diagnosis 不进入检索字段。

    text = "Text" 字段（病例正文），用于 embedding。
    diagnosis = Diagnosis 字段。
    """
    records: List[dict] = []
    # max_rows 全局上限优先于 max_per_split
    global_max = max_rows
    train_dir = os.path.normpath(train_dir)
    test_dir = os.path.normpath(test_dir)

    split_dirs = [("train", train_dir)]
    if include_test:
        split_dirs.append(("test", test_dir))

    for split_label, folder in split_dirs:
        if not os.path.isdir(folder):
            print(f"[WARN] ddxplus_cases: 目录不存在 -> {folder}")
            continue

        files = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".json") and os.path.isfile(os.path.join(folder, f))
            ],
            key=_natural_sort_key,
        )

        if max_per_split is not None:
            files = files[:max_per_split]

        # 全局上限
        remaining = None
        if global_max is not None:
            remaining = global_max - len(records)
            if remaining <= 0:
                break
            files = files[:remaining]

        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    case = json.load(f)
            except Exception as e:
                print(f"[WARN] ddxplus_cases: 跳过 {fpath}，读取失败: {e}")
                continue

            participant_no = case.get("Participant No.", os.path.basename(fpath))
            text = _clean_text(case.get("Text") or case.get("Symptoms") or "")
            if split_label == "test":
                text = re.sub(r"\n?Diagnosis:\s*.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
            diagnosis = str(case.get("Diagnosis") or case.get("Processed Diagnosis") or "").strip()

            record = {
                "id": f"ddxplus_cases:{split_label}:{participant_no}",
                "source": "ddxplus_cases",
                "title": f"{diagnosis or 'Unknown'} case, {case.get('Age', '?')}{case.get('Sex', '?')}".strip(),
                "text": text,
                "diagnosis": diagnosis,
                "metadata": {
                    "participant_no": participant_no,
                    "age": str(case.get("Age", "")),
                    "sex": str(case.get("Sex", "")),
                    "level_1": str(case.get("Level 1", "")),
                    "level_2": str(case.get("Level 2", "")),
                    "differential_diagnosis": str(case.get("Differential Diagnosis", "")),
                    "split": split_label,
                },
            }
            records.append(record)

    print(f"[INFO] ddxplus_cases: {len(records)} records loaded")
    return records


# ============================================================
# 2. ddxplus_kg
# ============================================================


def adapt_ddxplus_kg(
    kg_path: str = "./dataset/knowledge graph of DDXPlus.xlsx",
    max_rows: Optional[int] = None,
) -> List[dict]:
    """
    读取 DDXPlus 知识图谱 Excel 文件。
    每条三元组展开为一个 embedding 文本："subject relation object"。
    diagnosis 为空（KG 三元组无诊断标签）。
    """
    import pandas as pd

    if not os.path.exists(kg_path):
        print(f"[WARN] ddxplus_kg: KG 文件不存在 -> {kg_path}")
        return []

    try:
        df = pd.read_excel(kg_path, usecols=["subject", "relation", "object"])
    except Exception as e:
        print(f"[WARN] ddxplus_kg: 读取 Excel 失败 -> {e}")
        return []

    if max_rows is not None:
        df = df.head(max_rows)

    records: List[dict] = []
    seen: set = set()

    for idx, row in df.iterrows():
        subject = str(row.get("subject", "")).strip()
        relation = str(row.get("relation", "")).replace("_", " ").strip()
        obj = str(row.get("object", "")).strip()

        if not subject or not relation or not obj:
            continue

        text = f"{subject} {relation} {obj}"
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)

        records.append({
            "id": f"ddxplus_kg:{idx}",
            "source": "ddxplus_kg",
            "title": f"{subject} --{relation}--> {obj}",
            "text": text,
            "diagnosis": "",
            "metadata": {
                "subject": subject,
                "relation": relation,
                "object": obj,
            },
        })

    print(f"[INFO] ddxplus_kg: {len(records)} records loaded")
    return records


# ============================================================
# 3. pmc_patients
# ============================================================


def adapt_pmc_patients(
    csv_path: str = "./external_datasets/pmc_patients/PMC-Patients.csv",
    max_rows: Optional[int] = None,
) -> List[dict]:
    """
    读取 PMC-Patients CSV。
    字段：patient_id, patient_uid, PMID, file_path, title, patient, age, gender
    text = patient 字段（临床叙述），title = title 字段，diagnosis 无。
    如果列名不对，自动打印可用列名。
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"[WARN] pmc_patients: CSV 文件不存在 -> {csv_path}")
        print("[WARN] pmc_patients: 请先从 PMC-Patients 仓库下载数据放到 external_datasets/pmc_patients/")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] pmc_patients: 读取 CSV 失败 -> {e}")
        return []

    columns = df.columns.tolist()
    print(f"[INFO] pmc_patients: 可用列名: {columns}")

    # 尝试找到关键列
    text_col = None
    for candidate in ["patient", "summary", "text", "description", "abstract", "case", "narrative"]:
        if candidate in columns:
            text_col = candidate
            break
    if text_col is None:
        # 宽松匹配
        for c in columns:
            if any(kw in str(c).lower() for kw in ["patient", "summary", "text", "desc", "abstract", "case"]):
                text_col = c
                break

    title_col = None
    for candidate in ["title", "Title"]:
        if candidate in columns:
            title_col = candidate
            break

    id_col = None
    for candidate in ["patient_id", "patient_uid", "PMID", "id", "ID"]:
        if candidate in columns:
            id_col = candidate
            break

    if text_col is None:
        print(f"[WARN] pmc_patients: 找不到 patient/summary/text 类字段，可用列: {columns}")
        # 尝试打印前 3 行让用户判断
        print("[INFO] pmc_patients: 前 3 行预览：")
        print(df.head(3).to_string())
        return []

    if max_rows is not None:
        df = df.head(max_rows)

    records: List[dict] = []
    for idx, row in df.iterrows():
        text = _clean_text(row.get(text_col, ""))

        title = str(row.get(title_col, "")) if title_col else ""
        if not title:
            title = text[:80] + "..." if len(text) > 80 else text

        pid = str(row.get(id_col, idx)) if id_col else str(idx)

        # 提取年龄
        age_raw = row.get("age", "") if "age" in columns else ""
        try:
            if isinstance(age_raw, str) and age_raw:
                parsed = ast.literal_eval(age_raw)
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], list):
                    age_val = str(parsed[0][0])
                else:
                    age_val = str(age_raw)
            else:
                age_val = str(age_raw or "")
        except Exception:
            age_val = str(age_raw or "")

        records.append({
            "id": f"pmc_patients:{pid}",
            "source": "pmc_patients",
            "title": title,
            "text": text,
            "diagnosis": "",
            "metadata": {
                "patient_id": pid,
                "gender": str(row.get("gender", "")),
                "age": age_val,
                "PMID": str(row.get("PMID", "")),
            },
        })

    print(f"[INFO] pmc_patients: {len(records)} records loaded (text_col={text_col})")
    return records


# ============================================================
# 4. medcase_reasoning
# ============================================================


def adapt_medcase_reasoning(
    csv_path: str = "./external_datasets/medcase_reasoning/medcasereasoning_core.csv",
    max_rows: Optional[int] = None,
) -> List[dict]:
    """
    读取 medcasereasoning_core.csv（14,489 行）。
    字段：pmcid, title, journal, article_link, publication_date, text,
          case_prompt, diagnostic_reasoning, final_diagnosis, split
    text = case_prompt（患者主诉/摘要），diagnosis = final_diagnosis。
    """
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"[WARN] medcase_reasoning: CSV 文件不存在 -> {csv_path}")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] medcase_reasoning: 读取 CSV 失败 -> {e}")
        return []

    # 去掉 Unnamed 列
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    columns = df.columns.tolist()
    print(f"[INFO] medcase_reasoning: 可用列名: {columns}")

    text_col = None
    for candidate in ["case_prompt", "text", "summary", "patient", "abstract", "case", "description", "reasoning"]:
        if candidate in columns:
            text_col = candidate
            break
    if text_col is None:
        for c in columns:
            if any(kw in str(c).lower() for kw in ["prompt", "case", "text", "summary", "patient", "abstract"]):
                text_col = c
                break

    diag_col = None
    for candidate in ["final_diagnosis", "diagnosis", "label", "condition", "pathology"]:
        if candidate in columns:
            diag_col = candidate
            break

    title_col = None
    for candidate in ["title", "Title"]:
        if candidate in columns:
            title_col = candidate
            break

    if text_col is None:
        print(f"[WARN] medcase_reasoning: 找不到文本字段，可用列: {columns}")
        return []

    if max_rows is not None:
        df = df.head(max_rows)

    records: List[dict] = []
    for idx, row in df.iterrows():
        text = _clean_text(row.get(text_col, ""))
        if not text:
            continue

        diagnosis = str(row.get(diag_col, "")) if diag_col else ""
        title = str(row.get(title_col, "")) if title_col else text[:80]

        records.append({
            "id": f"medcase_reasoning:{idx}",
            "source": "medcase_reasoning",
            "title": title,
            "text": text,
            "diagnosis": diagnosis,
            "metadata": {
                "pmcid": str(row.get("pmcid", "")),
                "journal": str(row.get("journal", "")),
                "publication_date": str(row.get("publication_date", "")),
                "split": str(row.get("split", "")),
                "diagnostic_reasoning": str(row.get("diagnostic_reasoning", ""))[:500] if "diagnostic_reasoning" in columns else "",
            },
        })

    print(f"[INFO] medcase_reasoning: {len(records)} records loaded (text_col={text_col}, diag_col={diag_col})")
    return records


# ============================================================
# 5. open_patients
# ============================================================


def adapt_open_patients(
    jsonl_path: str = "./external_datasets/open_patients/Open-Patients.jsonl",
    max_rows: Optional[int] = None,
) -> List[dict]:
    """
    读取 Open-Patients.jsonl（每行一个 JSON，约 180k 行）。
    字段：_id（来源标识），description（临床笔记）。
    text = description，diagnosis 无。
    逐行读取，不全部加载到内存。
    """
    if not os.path.exists(jsonl_path):
        print(f"[WARN] open_patients: JSONL 文件不存在 -> {jsonl_path}")
        return []

    records: List[dict] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                doc_id = str(obj.get("_id", f"open_patients:{idx}"))
                description = str(obj.get("description", ""))

                text = _clean_text(description)
                if not text:
                    continue

                records.append({
                    "id": f"open_patients:{doc_id}",
                    "source": "open_patients",
                    "title": doc_id,
                    "text": text,
                    "diagnosis": "",
                    "metadata": {
                        "source_id": doc_id,
                    },
                })

                if max_rows is not None and len(records) >= max_rows:
                    break
    except Exception as e:
        print(f"[WARN] open_patients: 读取 JSONL 失败 -> {e}")

    print(f"[INFO] open_patients: {len(records)} records loaded")
    return records


# ============================================================
# 6. multicare_repo
# ============================================================


def adapt_multicare_repo(
    repo_dir: str = "./external_datasets/multicare_repo",
) -> List[dict]:
    """
    MultiCaRe 临床文本托管在 Zenodo（DOI: 10.5281/zenodo.10079369）。
    本仓库只包含分类学元数据和图像标签，没有临床叙述文本。
    如需 MultiCaRe 数据，请通过 multiversity 库从 Zenodo 下载。
    当前返回 []，不阻塞其他向量库构建。
    """
    repo_dir = os.path.normpath(repo_dir)

    if not os.path.isdir(repo_dir):
        print(f"[WARN] multicare_repo: 目录不存在 -> {repo_dir}")
        return []

    print("[WARN] multicare_repo: MultiCaRe 临床叙述文本托管在 Zenodo.")
    print("[WARN] multicare_repo: DOI: 10.5281/zenodo.10079369")
    print("[WARN] multicare_repo: 可以通过 multiversity Python 库下载：pip install multiversity")
    print("[WARN] multicare_repo: 本地仓库只包含分类学统计和图像标签，暂无可直接嵌入的文本。")
    print("[WARN] multicare_repo: 当前返回 0 条记录，不会构建该向量库。")

    return []


# ============================================================
# 7. synthea_records
# ============================================================


def adapt_synthea_records(
    repo_dir: str = "./external_datasets/synthea",
) -> List[dict]:
    """
    Synthea 是一个 Java 项目（患者数据生成器），本地无预生成的输出数据。
    需要先运行 Synthea 生成 FHIR/CSV 输出，再将输出路径传给此适配器。
    当前返回 []，不阻塞其他向量库构建。
    """
    repo_dir = os.path.normpath(repo_dir)

    if not os.path.isdir(repo_dir):
        print(f"[WARN] synthea_records: 目录不存在 -> {repo_dir}")
        return []

    output_dir = os.path.join(repo_dir, "output")
    if not os.path.isdir(output_dir):
        print("[WARN] synthea_records: 未找到 output/ 目录，尚未生成模拟患者数据。")
        print("[WARN] synthea_records: Synthea 是一个 Java 项目，需要先运行生成器：")
        print("[WARN] synthea_records:   1. 安装 JDK 17+")
        print("[WARN] synthea_records:   2. cd synthea && ./run_synthea -p 1000")
        print("[WARN] synthea_records:   3. 生成后将 output/csv/ 或 output/fhir/ 下的数据传入本适配器")
        print("[WARN] synthea_records: 当前返回 0 条记录，不会构建该向量库。")
        return []

    # 尝试读取 FHIR NDJSON
    fhir_dir = os.path.join(output_dir, "fhir")
    records: List[dict] = []
    if os.path.isdir(fhir_dir):
        for fname in os.listdir(fhir_dir):
            if fname.endswith(".json") or fname.endswith(".ndjson"):
                fpath = os.path.join(fhir_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                bundle = json.loads(line)
                            except json.JSONDecodeError:
                                # 普通 JSON 文件
                                bundle = json.loads(line)
                            # 从 FHIR Bundle 提取患者信息
                            if isinstance(bundle, dict) and bundle.get("resourceType") == "Bundle":
                                entries = bundle.get("entry", [])
                                patient_text_parts = []
                                for entry in entries:
                                    resource = entry.get("resource", {})
                                    rtype = resource.get("resourceType", "")
                                    if rtype == "Patient":
                                        name_list = resource.get("name", [])
                                        name_str = " ".join(
                                            " ".join(n.get("given", [])) + " " + n.get("family", "")
                                            for n in name_list
                                        ).strip()
                                        patient_text_parts.append(f"Patient: {name_str}")
                                    elif rtype == "Condition":
                                        code = resource.get("code", {}).get("text", "")
                                        if code:
                                            patient_text_parts.append(f"Condition: {code}")
                                    elif rtype == "MedicationStatement":
                                        med = resource.get("medicationCodeableConcept", {}).get("text", "")
                                        if med:
                                            patient_text_parts.append(f"Medication: {med}")
                                    elif rtype == "Observation":
                                        obs_code = resource.get("code", {}).get("text", "")
                                        obs_value = resource.get("valueQuantity", {}).get("value", "")
                                        if obs_code:
                                            patient_text_parts.append(f"Observation: {obs_code} = {obs_value}")

                                text = "\n".join(patient_text_parts[:20])
                                if text:
                                    records.append({
                                        "id": f"synthea_records:{fname}:{len(records)}",
                                        "source": "synthea_records",
                                        "title": f"FHIR patient {len(records)+1}",
                                        "text": text,
                                        "diagnosis": "",
                                        "metadata": {"source_file": fname},
                                    })
                except Exception as e:
                    print(f"[WARN] synthea_records: 跳过 {fpath}，读取失败: {e}")

    if not records:
        print("[WARN] synthea_records: 未找到可解析的 FHIR 患者数据。")
        print("[WARN] synthea_records: 请先运行 ./run_synthea -p 1000 生成数据。")
    else:
        print(f"[INFO] synthea_records: {len(records)} records loaded from FHIR output")

    return records


# ============================================================
# 适配器映射（供 scripts 使用）
# ============================================================

ADAPTER_MAP: Dict[str, Any] = {
    "ddxplus_cases": adapt_ddxplus_cases,
    "ddxplus_kg": adapt_ddxplus_kg,
    "pmc_patients": adapt_pmc_patients,
    "medcase_reasoning": adapt_medcase_reasoning,
    "open_patients": adapt_open_patients,
    "multicare_cases": adapt_multicare_repo,
    "synthea_records": adapt_synthea_records,
}
