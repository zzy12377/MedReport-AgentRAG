# -*- coding: utf-8 -*-
"""
main_MedRAG.py

修复点：
1. 训练集路径使用 authentication.py 的 ob_path，不再写死错误路径。
2. embedding 先走 SiliconFlow；失败后自动回退本地 sentence-transformers。
3. document/query embedding 使用同一个后端，避免维度不一致。
4. embedding 缓存加 meta，避免每次重新生成，也方便判断实际模型。
5. level_3_to_level_2 从 dataset/AI Data Set with Categories.csv 自动读取。
6. KG 检索同时兼容：
   - KG_Retrieve 返回疾病名；
   - KG_Retrieve 返回 respiratory_system 这类系统分类名。
7. system prompt 改成 DDXPlus 通用辅助诊断，不再是 pain management。
"""

import os
import re
import json
import warnings
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

from KG_Retrieve import main_get_category_and_level3
from authentication import (
    api_key,
    base_url,
    chat_model,
    embedding_model,
    ob_path,
    ground_truth_file_path,
)

from embedding_backend import (
    get_embedding_state,
    set_embedding_state,
    set_embedding_state_from_meta,
    safe_model_name,
    local_sentence_transformer_embeddings,
)


client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url,
)

# 记录当前 embedding 实际使用的后端，保证 document/query embedding 维度一致。
set_embedding_state(backend="remote", model=embedding_model)


safe_embedding_model_name = safe_model_name(embedding_model)
document_embeddings_file_path = f"./dataset/document_embeddings_{safe_embedding_model_name}.npy"
document_embeddings_meta_path = f"./dataset/document_embeddings_{safe_embedding_model_name}.json"


def load_case_text(file_path: str) -> str:
    """
    读取 participant_x.json 的病例正文。
    注意：embedding 应该嵌入病例内容，而不是嵌入文件路径。
    """
    with open(file_path, "r", encoding="utf-8") as f:
        patient_case = json.load(f)

    if isinstance(patient_case, dict):
        if patient_case.get("Text"):
            return str(patient_case["Text"])

        parts = []
        for key in [
            "Participant No.",
            "Age",
            "Sex",
            "Symptoms",
            "Differential Diagnosis",
            "Diagnosis",
            "Processed Diagnosis",
            "Level 2",
            "Level 1",
            "Pain Presentation and Description",
            "Pain descriptions and assorted symptoms (self-report)",
            "Pain restriction",
        ]:
            if key in patient_case and patient_case[key]:
                parts.append(f"{key}: {patient_case[key]}")
        return "\n".join(parts)

    return str(patient_case)


def _local_sentence_transformer_embeddings(texts: List[str], reason: str = "SiliconFlow embeddings 接口调用失败") -> np.ndarray:
    """
    SiliconFlow embedding 接口不可用时，自动下载并使用本地 sentence-transformers。
    默认模型 BAAI/bge-small-en-v1.5，CPU 可跑，内存远低于 16GB。
    """
    return local_sentence_transformer_embeddings(texts, reason=reason)


def _remote_siliconflow_embeddings(texts: List[str]) -> np.ndarray:
    embeddings = []
    for text in tqdm(texts):
        response = client.embeddings.create(
            input=text,
            model=embedding_model,
        )
        embeddings.append(response.data[0].embedding)

    vectors = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)
    set_embedding_state(backend="remote", model=embedding_model, dim=int(vectors.shape[1]))
    return vectors


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    默认使用 SiliconFlow embeddings。
    如果接口报错，自动回退到本地 sentence-transformers。
    如果已经加载过本地缓存，则 query 也会使用同一个本地模型，避免维度不一致。
    """
    if isinstance(texts, str):
        texts = [texts]

    texts = [str(t) for t in texts]

    state = get_embedding_state()
    if state.get("backend") == "local":
        return _local_sentence_transformer_embeddings(
            texts,
            reason=f"already using local embedding model: {state.get('model')}",
        )

    try:
        return _remote_siliconflow_embeddings(texts)
    except Exception as e:
        print("[WARN] SiliconFlow embeddings 接口调用失败，准备自动下载并切换到本地 sentence-transformers。")
        print(f"[WARN] Remote embedding error: {e}")
        return _local_sentence_transformer_embeddings(texts, reason=f"remote embedding error: {e}")


def get_query_embedding(query: str) -> np.ndarray:
    return get_embeddings([query])[0]


def _rebuild_document_embeddings(reason: str) -> np.ndarray:
    """当 remote 缓存和 local query 维度不一致时，自动重建训练集 embedding。"""
    global document_embeddings

    if not documents:
        raise ValueError("训练集 documents 为空，无法重建 document embeddings。")

    print(f"[WARN] {reason}")
    print("[INFO] Rebuilding document embeddings with current embedding backend...")
    document_texts = [load_case_text(path) for path in documents]
    document_embeddings = get_embeddings(document_texts)
    save_embeddings(document_embeddings, document_embeddings_file_path)
    _write_embedding_meta(document_embeddings_meta_path, document_embeddings)
    print(
        f"[INFO] Rebuilt and saved document embeddings: {document_embeddings_file_path}, "
        f"shape={document_embeddings.shape}, backend={get_embedding_state().get('backend')}, "
        f"model={get_embedding_state().get('model')}"
    )
    return document_embeddings


def Faiss(document_embeddings_input: np.ndarray, query_embedding: np.ndarray, k: int):
    # 优先使用模块内最新的全局 document_embeddings。这样即使 main.py 里传入的是旧缓存，
    # 本函数在自动重建后也能继续使用新矩阵。
    global document_embeddings
    active_embeddings = globals().get("document_embeddings", document_embeddings_input)
    if active_embeddings is None or np.asarray(active_embeddings).size == 0:
        active_embeddings = document_embeddings_input

    document_matrix = np.asarray(active_embeddings, dtype="float32")

    if query_embedding.ndim == 1:
        query_embedding = np.array([query_embedding], dtype="float32")
    else:
        query_embedding = np.asarray(query_embedding, dtype="float32")

    if document_matrix.ndim != 2:
        raise ValueError(f"document_embeddings 维度错误：{document_matrix.shape}")

    if document_matrix.size == 0:
        raise ValueError("document_embeddings 为空，请先生成训练集并重新运行。")

    if query_embedding.shape[1] != document_matrix.shape[1]:
        document_matrix = _rebuild_document_embeddings(
            "query embedding 和 document embedding 维度不一致，可能是 API 失败后切到了本地模型。"
            f" query dim={query_embedding.shape[1]}, document dim={document_matrix.shape[1]}"
        )

        if query_embedding.shape[1] != document_matrix.shape[1]:
            raise ValueError(
                "自动重建后维度仍不一致。\n"
                f"query dim={query_embedding.shape[1]}, document dim={document_matrix.shape[1]}\n"
                "请删除 dataset\\document_embeddings*.npy 和 dataset\\document_embeddings*.json 后重试。"
            )

    faiss.normalize_L2(query_embedding)

    index = faiss.IndexFlatIP(document_matrix.shape[1])
    index.add(document_matrix)
    _, indices = index.search(query_embedding, k)
    print("index: ", indices)
    return indices


def extract_diagnosis(generated_text: str):
    diagnoses = re.findall(r"\*\*Diagnosis\*\*:\s(.*?)\n", generated_text)
    return diagnoses


def remove_parentheses(text: Any) -> str:
    return re.sub(r"\(.*?\)", "", str(text)).strip()


def _norm_diagnosis_name(value: Any) -> str:
    """统一疾病名 / 类别名，解决大小写、下划线、连字符不一致。"""
    text = str(value or "").strip()
    text = remove_parentheses(text)
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _norm_relation(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def KG_preprocess(file_path: str):
    kg_data = pd.read_excel(file_path, usecols=["subject", "relation", "object"])
    kg_data["subject"] = kg_data["subject"].apply(remove_parentheses)
    kg_data["object"] = kg_data["object"].apply(remove_parentheses)

    knowledge_graph = {}
    for _, row in kg_data.iterrows():
        subject = row["subject"]
        relation = row["relation"]
        obj = row["object"]

        if subject not in knowledge_graph:
            knowledge_graph[subject] = []
        knowledge_graph[subject].append((relation, obj))

        if obj not in knowledge_graph:
            knowledge_graph[obj] = []
        knowledge_graph[obj].append((relation, subject))
    return knowledge_graph


def extract_features_from_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        patient_case = json.load(file)

    symptoms = patient_case.get("Symptoms", "")
    diagnosis = patient_case.get("Diagnosis", "")
    return symptoms, diagnosis


def load_level_3_to_level_2_from_csv(csv_path: str) -> Dict[str, str]:
    """
    从 dataset/AI Data Set with Categories.csv 自动读取：
        Processed Diagnosis -> Level 2

    这样不再手写两个映射。
    """
    if not csv_path or not os.path.exists(csv_path):
        print(f"[WARN] 找不到诊断层级 CSV：{csv_path}")
        return {}

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] 读取诊断层级 CSV 失败：{csv_path}. Error: {e}")
        return {}

    required = {"Processed Diagnosis", "Level 2"}
    if not required.issubset(set(df.columns)):
        print(
            "[WARN] CSV 缺少 Processed Diagnosis 或 Level 2 列，"
            f"当前列：{list(df.columns)}"
        )
        return {}

    mapping: Dict[str, str] = {}
    for _, row in df[["Processed Diagnosis", "Level 2"]].dropna().iterrows():
        diagnosis = str(row["Processed Diagnosis"]).strip()
        level2 = str(row["Level 2"]).strip()
        if not diagnosis or not level2:
            continue

        # 同时存 raw / lower / normalized 三种 key，增强匹配鲁棒性。
        mapping[diagnosis] = level2
        mapping[diagnosis.lower()] = level2
        mapping[_norm_diagnosis_name(diagnosis)] = level2

    print(f"[INFO] Loaded diagnosis -> Level 2 mapping: {len(mapping)} keys")
    return mapping


level_3_to_level_2 = load_level_3_to_level_2_from_csv(ground_truth_file_path)


def _read_kg_file(kg_path: str) -> Optional[pd.DataFrame]:
    if not kg_path or not os.path.exists(kg_path):
        print(f"[WARN] KG 文件不存在：{kg_path}")
        return None

    try:
        kg_data = pd.read_excel(kg_path, usecols=["subject", "relation", "object"])
    except Exception as e:
        print(f"[WARN] 无法读取 KG 文件：{kg_path}. Error: {e}")
        return None

    for col in ["subject", "relation", "object"]:
        kg_data[col] = kg_data[col].astype(str).map(remove_parentheses)

    kg_data["_subject_norm"] = kg_data["subject"].map(_norm_diagnosis_name)
    kg_data["_relation_norm"] = kg_data["relation"].map(_norm_relation)
    kg_data["_object_norm"] = kg_data["object"].map(_norm_diagnosis_name)
    return kg_data


def _candidate_diagnoses_from_csv(level_2_value: str) -> List[str]:
    """从 CSV 映射里找 Level 2 对应的 Processed Diagnosis。"""
    if not level_3_to_level_2:
        return []

    target = _norm_diagnosis_name(level_2_value)
    candidates = []
    for diagnosis, level2 in level_3_to_level_2.items():
        if _norm_diagnosis_name(level2) == target:
            candidates.append(str(diagnosis))
    return candidates


def _candidate_diagnoses_from_kg(kg_data: pd.DataFrame, level_2_value: str) -> List[str]:
    """
    兼容两种情况：
    1. level_2_value 本身是疾病名，例如 GERD / Bronchitis；
    2. level_2_value 是系统分类名，例如 respiratory_system。

    对第 2 种情况，从 KG 中找：<disease> is a <respiratory_system>。
    """
    target = _norm_diagnosis_name(level_2_value)
    candidates = []

    # 情况 1：target 直接作为 subject 出现在 KG 中。
    direct_rows = kg_data[kg_data["_subject_norm"] == target]
    if not direct_rows.empty:
        candidates.extend(direct_rows["subject"].dropna().astype(str).tolist())

    # 情况 2：target 作为 object/category 出现，例如 bronchitis is a respiratory_system。
    category_rows = kg_data[
        (kg_data["_object_norm"] == target)
        & (kg_data["_relation_norm"].isin(["is a", "is", "belongs to", "part of"]))
    ]
    if not category_rows.empty:
        candidates.extend(category_rows["subject"].dropna().astype(str).tolist())

    # 去重但保留顺序。
    seen = set()
    unique = []
    for item in candidates:
        key = _norm_diagnosis_name(item)
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def get_additional_info_from_level_2(participant_no, kg_path: str, top_n: int, match_n: int):
    """
    根据 KG_Retrieve 返回的 Level 2 / category，找到对应诊断，再从 KG 里取相关关系。
    如果 KG 或映射不可用，返回 None，不阻塞主流程。
    """
    try:
        level_2_values = main_get_category_and_level3(match_n, participant_no, top_n)
    except Exception as e:
        print(f"[WARN] KG_Retrieve failed for Participant No. {participant_no}: {e}")
        return None

    if not level_2_values:
        print(f"No data found for Participant No.: {participant_no}")
        return None

    kg_data = _read_kg_file(kg_path)
    if kg_data is None or kg_data.empty:
        print("[WARN] KG data is empty or unavailable.")
        return None

    additional_info = []

    for level_2_value in level_2_values:
        level_2_value = str(level_2_value).strip()
        if not level_2_value:
            continue

        # 先从 CSV 找，再从 KG 找。
        relevant_diagnoses = []
        relevant_diagnoses.extend(_candidate_diagnoses_from_csv(level_2_value))
        relevant_diagnoses.extend(_candidate_diagnoses_from_kg(kg_data, level_2_value))

        # 最后兜底：把 level_2_value 当作疾病名本身。
        relevant_diagnoses.append(level_2_value)

        # 去重。
        seen = set()
        unique_diagnoses = []
        for diagnosis in relevant_diagnoses:
            key = _norm_diagnosis_name(diagnosis)
            if key and key not in seen:
                seen.add(key)
                unique_diagnoses.append(diagnosis)

        print(f"KG target: {level_2_value}")
        print("Relevant diagnoses:", unique_diagnoses[:15])

        # 每个候选疾病只取少量 KG 关系，避免第一个疾病占满 30 条，
        # 导致后面的 GERD / cardiac 等候选完全进不了 prompt。
        max_relations_per_diagnosis = 6
        preferred_relations = {
            "has_symptomatology",
            "has_anamnesis",
            "has_exposure",
            "has_lifestyle",
            "is_a",
        }

        for diagnosis in unique_diagnoses:
            diagnosis_norm = _norm_diagnosis_name(diagnosis)
            related_info = kg_data[kg_data["_subject_norm"] == diagnosis_norm].copy()

            if related_info.empty:
                continue

            related_info["_relation_priority"] = related_info["relation"].astype(str).map(
                lambda r: 0 if str(r) in preferred_relations else 1
            )
            related_info = related_info.sort_values("_relation_priority")

            used = 0
            seen_sentences = set()
            for _, row in related_info.iterrows():
                subject = str(row["subject"])
                relation = str(row["relation"]).replace("_", " ")
                obj = str(row["object"])
                sentence = f"{subject} {relation} {obj}"
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                additional_info.append(sentence)
                used += 1
                if used >= max_relations_per_diagnosis:
                    break

    if not additional_info:
        print("No additional information found.")
        return None

    # 总长度仍做限制，但由于上面已经按疾病均分，多个候选都能进入 prompt。
    final_info = ", ".join(additional_info[:36])
    print("Additional Info:", final_info)
    return final_info


def get_system_prompt_for_RAGKG():
    return """
You are a careful medical decision-support assistant for a DDXPlus-style differential diagnosis task.

You will receive:
1. A new synthetic patient case, including age, sex, symptoms, antecedents, and possible differential diagnoses.
2. Retrieved similar patient cases from the local case database.
3. Optional knowledge-graph information about relevant diseases.

Your goals:
- Identify the most likely diagnosis from the provided clinical information.
- Prefer diagnoses listed in the case's Differential Diagnosis field unless the evidence strongly supports another diagnosis.
- Explain the reasoning using symptoms, risk factors, differential diagnoses, and retrieved similar cases.
- Use knowledge-graph information only as supporting evidence; do not let a broad category override specific patient symptoms.
- Suggest what additional questions or observations would help distinguish the closest diagnoses.
- Keep the answer clinically cautious and avoid claiming certainty beyond the evidence.
- Do not invent patient facts not present in the case.
- This is a decision-support output for an academic software project, not a substitute for a licensed clinician.

Use this output format exactly:

### Diagnoses
1. **Diagnosis**: <most likely diagnosis>

### Explanation
1. **Clinical reasoning**: <why this diagnosis fits>
2. **Differential considerations**: <similar diagnoses and how to distinguish them>
3. **Use of retrieved cases / knowledge graph**: <how retrieved evidence influenced the answer>

### Instructive Questions
1. **Questions**: <comma-separated missing questions or observations>

### Suggested Evaluation
1. **Evaluation 1**: <recommended next evaluation or observation>
2. **Evaluation 2**: <optional>

### Safety Note
1. **Note**: This output is for educational decision support only and should be reviewed by a qualified medical professional.
"""


def generate_diagnosis_report(
    path: str,
    query: str,
    retrieved_documents: str,
    i,
    top_n: int,
    match_n: int,
    model: Optional[str] = None,
):
    system_prompt = get_system_prompt_for_RAGKG()
    additional_info = get_additional_info_from_level_2(i, path, top_n=top_n, match_n=match_n)

    prompt = (
        f"New Patient Case:\n{query}\n\n"
        f"Retrieved Similar Cases:\n{retrieved_documents or 'None'}\n\n"
        f"Knowledge Graph Information:\n{additional_info or 'None available'}\n\n"
        "Please complete the diagnosis-support task using the required format."
    )

    model_to_use = model or chat_model
    if str(model_to_use).startswith("gpt-"):
        model_to_use = chat_model

    response = client.chat.completions.create(
        model=model_to_use,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def save_results_to_csv(results, output_file: str):
    df = pd.DataFrame(
        results,
        columns=["Participant No.", "Generated Diagnosis", "True Diagnosis", "Original Diagnosis"],
    )
    df.to_csv(output_file, index=False, encoding="utf-8-sig")


def _list_participant_files(folder_path: str) -> List[str]:
    if not folder_path or not os.path.isdir(folder_path):
        return []

    return [
        os.path.join(folder_path, file_name)
        for file_name in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file_name))
        and file_name.endswith(".json")
    ]


def _natural_sort_key(path: str):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


folder_path = ob_path
documents = sorted(_list_participant_files(folder_path), key=_natural_sort_key)


def save_embeddings(embeddings: np.ndarray, file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.save(file_path, embeddings)


def _write_embedding_meta(file_path: str, embeddings: np.ndarray):
    meta = {
        "requested_remote_model": embedding_model,
        "actual_backend": get_embedding_state().get("backend"),
        "actual_model": get_embedding_state().get("model"),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_embedding_meta(file_path: str):
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        set_embedding_state_from_meta(meta)
        return meta
    except Exception as e:
        print(f"[WARN] Failed to load embedding metadata: {e}")
        return None


def load_embeddings(file_path: str):
    _load_embedding_meta(document_embeddings_meta_path)
    return np.load(file_path).astype("float32")


if os.path.exists(document_embeddings_file_path):
    document_embeddings = load_embeddings(document_embeddings_file_path)
    print(
        f"[INFO] Loaded document embeddings: {document_embeddings_file_path}, "
        f"shape={document_embeddings.shape}, backend={get_embedding_state().get('backend')}, "
        f"model={get_embedding_state().get('model')}"
    )
else:
    if not documents:
        document_embeddings = np.empty((0, 0), dtype="float32")
        warnings.warn(
            f"训练集目录为空或不存在：{folder_path}。"
            "请先运行 python scripts\\prepare_ddxplus_for_medrag.py 生成数据。"
        )
    else:
        print(f"[INFO] Building document embeddings for {len(documents)} training cases...")
        document_texts = [load_case_text(path) for path in documents]
        document_embeddings = get_embeddings(document_texts)
        save_embeddings(document_embeddings, document_embeddings_file_path)
        _write_embedding_meta(document_embeddings_meta_path, document_embeddings)
        print(
            f"[INFO] Saved document embeddings: {document_embeddings_file_path}, "
            f"shape={document_embeddings.shape}, backend={get_embedding_state().get('backend')}, "
            f"model={get_embedding_state().get('model')}"
        )
