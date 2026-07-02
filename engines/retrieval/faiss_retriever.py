# -*- coding: utf-8 -*-
"""Full DDXPlus FAISS similar-case retrieval for RAG baselines."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engines.retrieval.embedding_engine import (
    DEFAULT_EMBEDDINGS_PATH,
    DEFAULT_METADATA_PATH,
    EmbeddingEngine,
)


DATA_PREP_HINT = "请先运行：python scripts/prepare_ddxplus_for_medrag.py"
DEFAULT_INDEX_PATH = "./storage/indexes/ddxplus_cases.faiss"
DEFAULT_INDEX_METADATA_PATH = "./storage/indexes/ddxplus_cases_metadata.jsonl"


@dataclass
class CaseRecord:
    case_id: str
    diagnosis: str
    raw_text: str
    source_path: str = ""


def _natural_sort_key(path: str) -> List[Any]:
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def _read_case_text(path: str) -> Tuple[str, str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    case_id = str(data.get("Participant No.") or os.path.splitext(os.path.basename(path))[0])
    diagnosis = str(data.get("Diagnosis") or data.get("Processed Diagnosis") or "")
    if data.get("Text"):
        text = str(data["Text"])
    else:
        parts = []
        for key in ["Age", "Sex", "Symptoms", "Differential Diagnosis", "Diagnosis"]:
            if data.get(key):
                parts.append(f"{key}: {data[key]}")
        text = "\n".join(parts)
    return case_id, diagnosis, text


def load_case_records(train_dir: str = "./dataset/df/train") -> List[CaseRecord]:
    train_dir = os.path.normpath(train_dir)
    if not os.path.isdir(train_dir):
        print(f"[WARN] 训练集目录不存在：{train_dir}")
        print(DATA_PREP_HINT)
        return []

    paths = sorted(
        [
            os.path.join(train_dir, name)
            for name in os.listdir(train_dir)
            if name.endswith(".json") and os.path.isfile(os.path.join(train_dir, name))
        ],
        key=_natural_sort_key,
    )
    records = []
    for path in paths:
        try:
            case_id, diagnosis, text = _read_case_text(path)
            records.append(CaseRecord(case_id=case_id, diagnosis=diagnosis, raw_text=text, source_path=path))
        except Exception as exc:
            print(f"[WARN] 跳过病例 {path}: {exc}")
    return records


class FaissCaseRetriever:
    """Load or build a persistent FAISS index over all DDXPlus train cases."""

    def __init__(
        self,
        train_dir: str = "./dataset/df/train",
        index_path: str = DEFAULT_INDEX_PATH,
        metadata_path: str = DEFAULT_INDEX_METADATA_PATH,
        embeddings_path: str = DEFAULT_EMBEDDINGS_PATH,
        embedding_metadata_path: str = DEFAULT_METADATA_PATH,
        top_k: int = 3,
        force_local: bool = True,
        batch_size: int = 32,
        force_rebuild: bool = False,
    ):
        self.train_dir = os.path.normpath(train_dir)
        self.index_path = os.path.normpath(index_path)
        self.metadata_path = os.path.normpath(metadata_path)
        self.embeddings_path = os.path.normpath(embeddings_path)
        self.embedding_metadata_path = os.path.normpath(embedding_metadata_path)
        self.top_k = top_k
        self.embedding_engine = EmbeddingEngine(force_local=force_local, batch_size=batch_size)
        self.index = None
        self.records: List[CaseRecord] = []

        if not force_rebuild and self._load_saved_index():
            return
        self.build_index(force_rebuild=force_rebuild, batch_size=batch_size)

    def _load_metadata_records(self) -> List[CaseRecord]:
        rows: List[CaseRecord] = []
        if not os.path.exists(self.metadata_path):
            return rows
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(
                        CaseRecord(
                            case_id=str(obj.get("case_id", "")),
                            diagnosis=str(obj.get("diagnosis", "")),
                            raw_text=str(obj.get("raw_text", "")),
                            source_path=str(obj.get("source_path", "")),
                        )
                    )
                except Exception:
                    continue
        return rows

    def _load_saved_index(self) -> bool:
        if not (os.path.exists(self.index_path) and os.path.exists(self.metadata_path)):
            return False
        try:
            import faiss

            index = faiss.read_index(self.index_path)
            records = self._load_metadata_records()
            if not records:
                print("[WARN] FAISS metadata is empty; rebuilding index.")
                return False
            if index.ntotal != len(records):
                print(f"[WARN] FAISS index count {index.ntotal} != metadata count {len(records)}; rebuilding index.")
                return False
            if index.d <= 0:
                print("[WARN] FAISS index dimension is invalid; rebuilding index.")
                return False

            self.index = index
            self.records = records
            print(f"[INFO] Loaded FAISS case index: {self.index_path} ({index.ntotal} vectors, dim={index.d})")
            return True
        except Exception as exc:
            print(f"[WARN] Failed to load FAISS index; rebuilding. Error: {exc}")
            return False

    def build_index(self, force_rebuild: bool = False, batch_size: int = 32) -> None:
        self.records = load_case_records(self.train_dir)
        if not self.records:
            self.index = None
            return
        try:
            import faiss
        except Exception as exc:
            print(f"[WARN] faiss 不可用，无法构建检索索引：{exc}")
            self.index = None
            return

        record_dicts = [asdict(record) for record in self.records]
        vectors = self.embedding_engine.embed_records_with_cache(
            record_dicts,
            embeddings_path=self.embeddings_path,
            metadata_path=self.embedding_metadata_path,
            batch_size=batch_size,
            force_rebuild=force_rebuild,
        ).astype("float32")
        if vectors.ndim != 2 or vectors.shape[0] != len(self.records):
            print("[WARN] embedding 结果异常，无法构建 FAISS 索引")
            self.index = None
            return
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(int(vectors.shape[1]))
        self.index.add(vectors)
        self._save_index()

    def _save_index(self) -> None:
        if self.index is None:
            return
        try:
            import faiss

            os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
            os.makedirs(os.path.dirname(self.metadata_path) or ".", exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                for record in self.records:
                    f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            print(f"[INFO] Saved FAISS case index: {self.index_path}")
            print(f"[INFO] Saved FAISS metadata: {self.metadata_path}")
        except Exception as exc:
            print(f"[WARN] 保存 FAISS 索引失败：{exc}")

    def retrieve_similar_cases(self, query_text: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.index is None or not self.records:
            print(f"[WARN] 检索索引不可用。{DATA_PREP_HINT}")
            return []
        try:
            import faiss

            query_vec = self.embedding_engine.embed_query(query_text).astype("float32")
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)
            if query_vec.shape[1] != self.index.d:
                print(
                    f"[WARN] query embedding 维度 {query_vec.shape[1]} 与索引维度 {self.index.d} 不一致，"
                    "请用相同 embedding 模型重建索引。"
                )
                return []
            faiss.normalize_L2(query_vec)
            k = min(int(top_k or self.top_k), len(self.records))
            scores, indices = self.index.search(query_vec, k)
            return self._format_results(scores[0], indices[0])
        except Exception as exc:
            print(f"[WARN] FAISS 检索失败：{exc}")
            return []

    def search_vectors(self, query_vectors: np.ndarray, top_k: int) -> List[List[Dict[str, Any]]]:
        if self.index is None or not self.records:
            return []
        import faiss

        vectors = np.asarray(query_vectors, dtype="float32")
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.shape[1] != self.index.d:
            raise ValueError(f"query dim {vectors.shape[1]} != index dim {self.index.d}")
        faiss.normalize_L2(vectors)
        k = min(int(top_k), len(self.records))
        scores, indices = self.index.search(vectors, k)
        return [self._format_results(row_scores, row_indices) for row_scores, row_indices in zip(scores, indices)]

    def _format_results(self, scores: np.ndarray, indices: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.records):
                continue
            record = self.records[int(idx)]
            results.append(
                {
                    "case_id": record.case_id,
                    "diagnosis": record.diagnosis,
                    "similarity": float(score),
                    "raw_text": record.raw_text,
                }
            )
        return results


def retrieve_similar_cases(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    return FaissCaseRetriever(top_k=top_k).retrieve_similar_cases(query_text, top_k=top_k)

