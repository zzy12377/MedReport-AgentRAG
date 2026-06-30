import os
import re
import string
import time
import json
from functools import lru_cache

import openai
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

from authentication import (
    api_key,
    base_url,
    embedding_model,
    ground_truth_file_path,
    augmented_features_path,
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

KG_file_path = augmented_features_path
file_path = ground_truth_file_path
embedding_save_path = "./Embeddings_saved/DDXPlus_KG_embeddings"

# Keep embedding inputs short and clean to avoid gateway-side HTML 400 errors.
MAX_EMBEDDING_CHARS = 4096
EMBEDDING_RETRY_TIMES = 3


def _clean_text_for_embedding(text, max_chars=MAX_EMBEDDING_CHARS):
    text = str(text or "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text or "empty medical concept"


def _normalise_node(value):
    value = re.sub(r"\(.*?\)", "", str(value)).strip()
    return value


def preprocess_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = _normalise_node(text)
    text = text.replace("_", " ").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "ISO-8859-1", "gbk"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _require_file(path, hint):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在。{hint}")


def _embedding_cache_name(model_name):
    return f"KG_embeddings_{safe_model_name(model_name)}.npy"


def _embedding_meta_name(embeddings_path):
    return embeddings_path + ".json"


def _write_embedding_meta(meta_path, embeddings):
    meta = {
        "requested_remote_model": embedding_model,
        "actual_backend": get_embedding_state().get("backend"),
        "actual_model": get_embedding_state().get("model"),
        "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else None,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _load_embedding_meta(meta_path):
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        set_embedding_state_from_meta(meta)
        return meta
    except Exception as e:
        print(f"[WARN] Failed to load KG embedding metadata: {e}")
        return None


def _remote_get_embedding(text):
    safe_text = _clean_text_for_embedding(text)
    last_error = None
    for attempt in range(1, EMBEDDING_RETRY_TIMES + 1):
        try:
            response = client.embeddings.create(input=safe_text, model=embedding_model)
            vector = np.asarray(response.data[0].embedding, dtype="float32")
            set_embedding_state(backend="remote", model=embedding_model, dim=int(vector.shape[0]))
            return vector
        except Exception as e:
            last_error = e
            print(f"KG embedding API failed, attempt {attempt}/{EMBEDDING_RETRY_TIMES}: {e}")
            time.sleep(min(2 * attempt, 6))
    raise RuntimeError(f"KG embedding API failed after retries. Last error: {last_error}")


def _get_embeddings(texts):
    if isinstance(texts, str):
        texts = [texts]
    safe_texts = [_clean_text_for_embedding(t) for t in texts]

    state = get_embedding_state()
    if state.get("backend") == "local":
        return local_sentence_transformer_embeddings(
            safe_texts,
            reason=f"KG already using local embedding model: {state.get('model')}",
        )

    try:
        vectors = []
        for text in tqdm(safe_texts):
            vectors.append(_remote_get_embedding(text))
        return np.asarray(vectors, dtype="float32")
    except Exception as e:
        print("[WARN] KG embedding API 报错，准备自动下载并切换到本地 sentence-transformers。")
        print(f"[WARN] KG remote embedding error: {e}")
        return local_sentence_transformer_embeddings(
            safe_texts,
            reason=f"KG remote embedding error: {e}",
        )


def _get_embedding(text):
    return _get_embeddings([text])[0]


def get_symptom_embeddings(symptom_nodes, save_path, force_rebuild=False):
    os.makedirs(save_path, exist_ok=True)
    embeddings_path = os.path.join(save_path, _embedding_cache_name(embedding_model))
    meta_path = _embedding_meta_name(embeddings_path)

    if os.path.exists(embeddings_path) and not force_rebuild:
        print("load existing KG embeddings...")
        _load_embedding_meta(meta_path)
        embeddings = np.load(embeddings_path).astype("float32")
        if len(embeddings) == len(symptom_nodes):
            return embeddings
        print("KG embedding cache size mismatch; regenerating...")

    print("generate new KG embeddings...")
    symptom_embeddings = _get_embeddings(symptom_nodes)
    symptom_embeddings = np.asarray(symptom_embeddings, dtype="float32")
    np.save(embeddings_path, symptom_embeddings)
    _write_embedding_meta(meta_path, symptom_embeddings)
    return symptom_embeddings


@lru_cache(maxsize=1)
def load_kg_resources():
    _require_file(KG_file_path, "请确认 dataset/knowledge graph of DDXPlus.xlsx 已存在。")

    kg_data = pd.read_excel(KG_file_path, usecols=["subject", "relation", "object"])
    kg_data["subject"] = kg_data["subject"].apply(_normalise_node)
    kg_data["object"] = kg_data["object"].apply(_normalise_node)

    knowledge_graph = {}
    for _, row in kg_data.iterrows():
        subject = row["subject"]
        relation = row["relation"]
        obj = row["object"]

        knowledge_graph.setdefault(subject, []).append((relation, obj))
        knowledge_graph.setdefault(obj, []).append((relation, subject))

    kg_data["object_preprocessed"] = kg_data.apply(
        lambda row: preprocess_text(row["object"]) if row["relation"] != "is_a" else None,
        axis=1,
    )
    symptom_nodes = kg_data["object_preprocessed"].dropna().unique().tolist()
    symptom_embeddings = get_symptom_embeddings(symptom_nodes, embedding_save_path)

    graph = nx.Graph()
    for node, edges in knowledge_graph.items():
        for relation, neighbor in edges:
            graph.add_edge(node, neighbor, relation=relation)

    return kg_data, knowledge_graph, symptom_nodes, symptom_embeddings, graph


def load_categories(data=None, kg_data=None):
    categories = []
    if data is not None and "Level 2" in data.columns:
        categories.extend(
            data["Level 2"].dropna().astype(str).str.split(",").explode().str.strip().tolist()
        )

    if kg_data is not None and {"relation", "subject", "object"}.issubset(kg_data.columns):
        # DDXPlus KG: disease --is_a--> Level 2/category-like nodes
        is_a_objects = kg_data.loc[kg_data["relation"] == "is_a", "object"].dropna().astype(str).tolist()
        categories.extend([x.strip() for x in is_a_objects])

    categories = [c for c in dict.fromkeys(categories) if c and c.lower() != "nan"]
    return categories


def find_top_n_similar_symptoms(query, symptom_nodes, symptom_embeddings, n):
    if pd.isna(query) or not str(query).strip():
        return []

    query_preprocessed = preprocess_text(query)
    query_embedding = np.asarray(_get_embedding(query_preprocessed), dtype="float32")
    if query_embedding.size == 0:
        return []

    if len(symptom_embeddings) > len(symptom_nodes):
        symptom_embeddings = symptom_embeddings[: len(symptom_nodes)]

    if symptom_embeddings.ndim != 2 or query_embedding.shape[0] != symptom_embeddings.shape[1]:
        print(
            "[WARN] KG query embedding 和 KG symptom embedding 维度不一致，"
            "可能是 API 失败后切到了本地模型，正在自动重建 KG embeddings..."
        )
        symptom_embeddings = get_symptom_embeddings(symptom_nodes, embedding_save_path, force_rebuild=True)

    if symptom_embeddings.ndim != 2 or query_embedding.shape[0] != symptom_embeddings.shape[1]:
        print(
            "[WARN] KG embeddings 自动重建后维度仍不一致，跳过本次 KG 相似症状检索。"
            f" query dim={query_embedding.shape[0]}, symptom dim="
            f"{symptom_embeddings.shape[1] if getattr(symptom_embeddings, 'ndim', 0) == 2 else 'invalid'}"
        )
        return []

    similarities = cosine_similarity([query_embedding], symptom_embeddings).flatten()

    top_n_symptoms = []
    unique_symptoms = set()
    top_n_indices = similarities.argsort()[::-1]

    for i in top_n_indices:
        if similarities[i] > 0.5 and symptom_nodes[i] not in unique_symptoms:
            top_n_symptoms.append(symptom_nodes[i])
            unique_symptoms.add(symptom_nodes[i])
        if len(top_n_symptoms) == n:
            break

    return top_n_symptoms


def get_diagnoses_for_symptom(symptom, graph):
    diagnoses = []
    if symptom in graph:
        for neighbor in graph.neighbors(symptom):
            edge_data = graph.get_edge_data(neighbor, symptom)
            if edge_data and edge_data.get("relation") != "is_a":
                diagnoses.append(neighbor)
    return diagnoses


def find_closest_category(top_symptoms, categories, top_n, graph):
    if not top_symptoms:
        print("Warning: top_symptoms is empty.")
        return []

    categories = [c for c in categories if c in graph]
    if not categories:
        print("Warning: no valid categories found in graph.")
        return []

    category_votes = {category: 0 for category in categories}

    for symptom in list(set(top_symptoms)):
        if symptom not in graph:
            print(f"Symptom node not found in graph: {symptom}")
            continue

        diagnosis_nodes = get_diagnoses_for_symptom(symptom, graph)
        for diagnosis in diagnosis_nodes:
            individual_diagnoses = str(diagnosis).split(",")
            for single_diagnosis in individual_diagnoses:
                single_diagnosis = single_diagnosis.strip().replace(" ", "_").lower()
                if single_diagnosis not in graph:
                    print(f"Diagnosis node not found in graph: {single_diagnosis}")
                    continue

                min_distance = float("inf")
                closest_category = None
                for category in categories:
                    try:
                        distance = nx.shortest_path_length(graph, source=single_diagnosis, target=category)
                    except nx.NetworkXNoPath:
                        distance = float("inf")

                    if distance < min_distance:
                        min_distance = distance
                        closest_category = category

                if closest_category:
                    category_votes[closest_category] += 1

    print("Category votes:", category_votes)
    sorted_categories = sorted(category_votes.items(), key=lambda x: x[1], reverse=True)
    return [cat for cat, vote in sorted_categories[:top_n] if vote > 0] or [cat for cat, _ in sorted_categories[:top_n]]


def get_subjects_for_objects(objects, kg_data):
    subjects = []
    processed_objects = [str(obj).replace(" ", "_") for obj in objects]
    for obj in processed_objects:
        for _, row in kg_data.iterrows():
            if row["object"] == obj:
                subjects.append(row["subject"])
    return subjects


def _first_existing_value(row, keys):
    for key in keys:
        if key in row.index and not pd.isna(row[key]) and str(row[key]).strip():
            return str(row[key])
    return ""



def _parse_differential_diagnoses(text, max_candidates=8):
    """
    DDXPlus 的 Differential Diagnosis / Pain restriction 不是症状字段，不能拿去做
    symptom embedding 检索；否则会把疾病名列表当成症状，导致 KG category vote 偏到某个系统。

    返回格式：[(diagnosis_name, probability_or_None), ...]
    """
    if pd.isna(text) or not str(text).strip():
        return []

    candidates = []
    parts = [part.strip() for part in str(text).split(";") if part.strip()]
    for part in parts:
        match = re.match(r"^(.*?)\s*\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)\s*$", part)
        if match:
            name = match.group(1).strip()
            try:
                score = float(match.group(2))
            except Exception:
                score = None
        else:
            # 不是 Differential Diagnosis 列表时不要误拆普通症状文本。
            # DDXPlus 的候选疾病一般包含括号概率；没有概率的只接受短疾病名。
            if len(part) > 80 or ":" in part:
                continue
            name = part.strip()
            score = None

        if not name:
            continue
        candidates.append((name, score))

    # 去重：保留最高概率/首次顺序。
    seen = set()
    unique = []
    for name, score in candidates:
        key = re.sub(r"\s+", " ", name.replace("_", " ").lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, score))
        if len(unique) >= max_candidates:
            break
    return unique


def _candidate_names_from_differential(text, max_candidates=8):
    return [name for name, _ in _parse_differential_diagnoses(text, max_candidates=max_candidates)]


def _looks_like_differential_list(text):
    return len(_parse_differential_diagnoses(text, max_candidates=2)) > 0


def _build_patient_query_fields(row):
    # Compatible with both the original CPDD-style columns and DDXPlus converted columns.
    pain_location = _first_existing_value(
        row,
        [
            "Pain Presentation and Description",
            "Pain Presentation and Description Areas of pain as per physiotherapy input",
            "Symptoms",
            "Evidences",
            "Evidence",
            "Patient Text",
            "Text",
        ],
    )
    pain_symptoms = _first_existing_value(
        row,
        [
            "Pain descriptions and assorted symptoms (self-report)",
            "Pain descriptions and assorted symptoms (self-report) Associated symptoms include: parasthesia, numbness, weakness, tingling, pins and needles",
            "Antecedents",
            "Symptoms Text",
            "Text",
        ],
    )

    # 注意：DDXPlus 转换后的 Pain restriction 往往是 Differential Diagnosis 候选疾病列表，
    # 不是症状。这里仍返回它，但 main_get_category_and_level3 会识别并避免把它当症状向量检索。
    pain_restriction = _first_existing_value(
        row,
        [
            "Pain restriction",
            "Initial Evidence",
            "INITIAL_EVIDENCE",
        ],
    )
    differential_text = _first_existing_value(row, ["Differential Diagnosis"])
    if not pain_restriction and differential_text:
        pain_restriction = differential_text
    return pain_location, pain_symptoms, pain_restriction


def main_get_category_and_level3(n, participant_no, top_n):
    kg_data, _, symptom_nodes, symptom_embeddings, graph = load_kg_resources()
    _require_file(file_path, "请先运行数据转换脚本，生成 dataset/AI Data Set with Categories.csv。")

    data = _safe_read_csv(file_path)
    if "Participant No." not in data.columns:
        raise KeyError("AI Data Set with Categories.csv 缺少 Participant No. 列。")

    row = data.loc[data["Participant No."].astype(str) == str(participant_no)]
    if row.empty:
        print(f"Participant No. {participant_no} not found!")
        return []

    row = row.iloc[0]
    categories = load_categories(data, kg_data)

    level2_truth = _first_existing_value(row, ["Level 2"])
    level3_real = _first_existing_value(row, ["Processed Diagnosis", "PATHOLOGY", "Diagnosis"])
    pain_location, pain_symptoms, pain_restriction = _build_patient_query_fields(row)

    print(f"truth level2: {level2_truth}")
    print(f"truth level3: {level3_real}")
    print(f"patient field 1: {pain_location}")
    print(f"patient field 2: {pain_symptoms}")
    print(f"patient field 3: {pain_restriction}")

    differential_candidates = _candidate_names_from_differential(pain_restriction, max_candidates=8)
    if differential_candidates:
        print("DDXPlus differential candidates:", differential_candidates)

    def process_symptom_field(field_value):
        if pd.isna(field_value) or str(field_value).strip() == "":
            return []
        return find_top_n_similar_symptoms(field_value, symptom_nodes, symptom_embeddings, n)

    top_location_nodes = process_symptom_field(pain_location)
    top_symptom_nodes = process_symptom_field(pain_symptoms)

    # 修复：DDXPlus 的 Pain restriction / Differential Diagnosis 是候选疾病列表，
    # 不是症状描述。原 CPDD 项目可以把 pain_restriction 作为症状维度，
    # 但迁移到 DDXPlus 后继续这样做会把“Bronchitis (0.20); GERD (0.17)...”
    # 当成症状去检索 KG，进而把 KG target 错误放大到 respiratory_system。
    if _looks_like_differential_list(pain_restriction):
        top_restriction_nodes = []
    else:
        top_restriction_nodes = process_symptom_field(pain_restriction)

    top_location_original = kg_data.loc[
        kg_data["object_preprocessed"].isin(top_location_nodes), "object"
    ].drop_duplicates()
    top_symptom_original = kg_data.loc[
        kg_data["object_preprocessed"].isin(top_symptom_nodes), "object"
    ].drop_duplicates()
    top_restriction_original = kg_data.loc[
        kg_data["object_preprocessed"].isin(top_restriction_nodes), "object"
    ].drop_duplicates()

    symptom_based_categories = find_closest_category(
        list(top_location_original) + list(top_symptom_original) + list(top_restriction_original),
        categories,
        top_n,
        graph,
    )

    # 对 DDXPlus：优先返回候选疾病本身，而不是返回一个宽泛系统大类。
    # 这样 downstream 只会取候选疾病的 KG 信息，不会因为一次 category vote 偏差
    # 就展开 respiratory_system 下的一大批无关疾病。
    if differential_candidates:
        max_return = min(len(differential_candidates), max(int(top_n or 1), 6))
        selected = differential_candidates[:max_return]
        print("KG symptom-based categories:", symptom_based_categories)
        print("KG returned disease candidates:", selected)
        return selected

    return symptom_based_categories
