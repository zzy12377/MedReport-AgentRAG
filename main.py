# -*- coding: utf-8 -*-
"""
main.py

修复点：
1. 在导入 main_MedRAG 之前检查 dataset/df/train、dataset/df/test 是否存在，避免 import 阶段直接崩溃。
2. 日志按病例分隔，避免把 patient_id=1 的模型输出误看成 patient_id=2。
3. FAISS 返回下标后读取真正病例正文作为 retrieved documents。
"""

import os
import re
import sys
import json

from authentication import (
    ob_path,
    test_folder_path,
    ground_truth_file_path,
    augmented_features_path,
    chat_model,
)


def print_data_prepare_help():
    print("\n[DATA NOT READY] MedRAG 运行所需数据还没准备好。")
    print("请先确认 DDXPlus 原始文件已放在：")
    print("  dataset\\ddxplus_raw\\")
    print("然后运行：")
    print("  python scripts\\prepare_ddxplus_for_medrag.py")
    print("\n运行成功后应存在：")
    print("  dataset\\AI Data Set with Categories.csv")
    print("  dataset\\df\\train\\participant_1.json")
    print("  dataset\\df\\test\\participant_1.json\n")


def ensure_dir_has_json(folder_path: str, label: str) -> bool:
    if not os.path.isdir(folder_path):
        print(f"[ERROR] {label}目录不存在：{folder_path}")
        return False

    json_files = [
        file_name
        for file_name in os.listdir(folder_path)
        if file_name.endswith(".json")
        and os.path.isfile(os.path.join(folder_path, file_name))
    ]

    if not json_files:
        print(f"[ERROR] {label}目录存在，但里面没有 participant_*.json：{folder_path}")
        return False

    return True


def ensure_dataset_ready() -> bool:
    ok = True

    if not os.path.exists(ground_truth_file_path):
        print(f"[ERROR] 缺少总 CSV：{ground_truth_file_path}")
        ok = False

    if not ensure_dir_has_json(ob_path, "训练集"):
        ok = False

    if not ensure_dir_has_json(test_folder_path, "测试集"):
        ok = False

    if not os.path.exists(augmented_features_path):
        print(f"[WARN] KG 文件不存在：{augmented_features_path}")
        print("[WARN] 主流程仍可尝试运行，但知识图谱增强信息可能为空。")

    if not ok:
        print_data_prepare_help()

    return ok


def natural_sort_key(path: str):
    """让 participant_2.json 排在 participant_10.json 前面。"""
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_participant_no(file_path: str, case_data):
    """
    优先从 json 里读取 Participant No.；
    没有则从 participant_1.json 这种文件名中解析 1。
    """
    if isinstance(case_data, dict):
        for key in ["Participant No.", "Participant No", "participant_no", "id", "ID"]:
            if key in case_data and str(case_data[key]).strip():
                try:
                    return int(case_data[key])
                except Exception:
                    return case_data[key]

    name = os.path.basename(file_path)
    match = re.search(r"(\d+)", name)
    if match:
        return int(match.group(1))

    return name


def get_true_diagnosis(case_data):
    """兼容 DDXPlus / 原 MedRAG / 自己构造数据的不同字段名。"""
    if not isinstance(case_data, dict):
        return ""

    for key in [
        "Processed Diagnosis",
        "Diagnosis",
        "PATHOLOGY",
        "pathology",
        "condition",
        "disease",
        "True Diagnosis",
    ]:
        value = case_data.get(key)
        if value:
            return str(value)

    return ""


def build_retrieved_documents(indices, documents, load_case_text, max_chars_per_case: int = 1200):
    """FAISS 返回的是 documents 的下标，这里把相似病例正文拼成文本给 LLM。"""
    retrieved = []

    if indices is None or len(indices) == 0:
        return ""

    for rank, idx in enumerate(indices[0], start=1):
        idx = int(idx)
        if idx < 0 or idx >= len(documents):
            continue

        doc_path = documents[idx]
        text = load_case_text(doc_path).strip()
        if len(text) > max_chars_per_case:
            text = text[:max_chars_per_case] + "..."

        retrieved.append(
            f"[Similar Case {rank}]\n"
            f"File: {os.path.basename(doc_path)}\n"
            f"{text}"
        )

    return "\n\n".join(retrieved)


def run_medrag(
    top_k: int = 3,
    top_n: int = 1,
    match_n: int = 5,
    max_cases: int = 5,
    output_file: str | None = None,
) -> int:
    """
    top_k: FAISS 检索相似病例数量。
    top_n / match_n: 传给 KG_Retrieve 的参数。
    max_cases: 先跑少量样本，避免 API 额度消耗过快。
    """
    if not ensure_dataset_ready():
        return 1

    # 数据检查通过后再导入 main_MedRAG。
    # 这样如果 dataset/df/train 不存在，不会在 import 阶段直接崩溃。
    from main_MedRAG import (
        documents,
        document_embeddings,
        load_case_text,
        get_query_embedding,
        Faiss,
        generate_diagnosis_report,
        save_results_to_csv,
    )

    test_files = [
        os.path.join(test_folder_path, file_name)
        for file_name in os.listdir(test_folder_path)
        if file_name.endswith(".json")
        and os.path.isfile(os.path.join(test_folder_path, file_name))
    ]
    test_files = sorted(test_files, key=natural_sort_key)

    if not test_files:
        print(f"[ERROR] 测试集目录为空：{test_folder_path}")
        print_data_prepare_help()
        return 1

    if not documents:
        print(f"[ERROR] 训练集 documents 为空。请检查：{ob_path}")
        print_data_prepare_help()
        return 1

    if document_embeddings.size == 0:
        print("[ERROR] document_embeddings 为空。请删除旧缓存后重试：")
        print("  del dataset\\document_embeddings*.npy")
        print("  del dataset\\document_embeddings*.json")
        return 1

    if output_file is None:
        output_file = f"./test_results_medrag_topk{top_k}_topn{top_n}_matchn{match_n}_cases{max_cases}.csv"

    results = []
    total = min(max_cases, len(test_files))

    for case_index, test_path in enumerate(test_files[:max_cases], start=1):
        print("\n" + "=" * 90)
        print(f"[CASE START] {case_index}/{total}")
        print(f"Test file: {test_path}")

        case_data = read_json(test_path)
        participant_no = get_participant_no(test_path, case_data)
        true_diagnosis = get_true_diagnosis(case_data)
        query = load_case_text(test_path)

        print(f"Participant No.: {participant_no}")
        print(f"True Diagnosis: {true_diagnosis}")
        print("-" * 90)

        query_embedding = get_query_embedding(query)
        indices = Faiss(document_embeddings, query_embedding, top_k)
        retrieved_documents = build_retrieved_documents(indices, documents, load_case_text)

        generated_report = generate_diagnosis_report(
            path=augmented_features_path,
            query=query,
            retrieved_documents=retrieved_documents,
            i=participant_no,
            top_n=top_n,
            match_n=match_n,
            model=chat_model,
        )

        results.append([
            participant_no,
            generated_report,
            true_diagnosis,
            true_diagnosis,
        ])

        print("\nGenerated report:")
        print(generated_report)
        print(f"[CASE END] Participant No.: {participant_no}")
        print("=" * 90)

    save_results_to_csv(results, output_file)
    print(f"\nDone. Results saved to: {output_file}")
    return 0


if __name__ == "__main__":
    # 先跑 5 条，确认流程没问题；之后再把 max_cases 改大。
    sys.exit(run_medrag(
        top_k=3,
        top_n=1,
        match_n=5,
        max_cases=5,
    ))
