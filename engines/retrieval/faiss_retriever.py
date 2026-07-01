# -*- coding: utf-8 -*-
"""FAISS similar-case retrieval for the phase-1 RAG baseline."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from engines.retrieval.embedding_engine import EmbeddingEngine


DATA_PREP_HINT = "请先运行：python scripts/prepare_ddxplus_for_medrag.py"


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


class FaissCaseRetriever:
    """Load or build a FAISS index over DDXPlus training cases."""

    def __init__(
        self,
        train_dir: str = "./dataset/df/train",
        index_dir: str = "./storage/knowledge/ddxplus_cases",
        vector_db_dir: str = "./vector_db/ddxplus_cases",
        top_k: int = 3,
        force_local: bool = True,
        prefer_vector_db: bool = True,
    ):
        self.train_dir = os.path.normpath(train_dir)
        self.index_dir = os.path.normpath(index_dir)
        self.vector_db_dir = os.path.normpath(vector_db_dir)
        self.top_k = top_k
        self.embedding_engine = EmbeddingEngine(force_local=force_local)
        self.index = None
        self.records: List[CaseRecord] = []

        if prefer_vector_db and self._load_vector_store(self.vector_db_dir):
            return
        if self._load_saved_index(self.index_dir):
            return
        self.build_index()

    def _load_vector_store(self, store_dir: str) -> bool:
        index_path = os.path.join(store_dir, "index.faiss")
        meta_path = os.path.join(store_dir, "meta.jsonl")
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False
        try:
            import faiss

            self.index = faiss.read_index(index_path)
            records: List[CaseRecord] = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    records.append(
                        CaseRecord(
                            case_id=str(obj.get("id") or obj.get("metadata", {}).get("participant_no") or ""),
                            diagnosis=str(obj.get("diagnosis") or ""),
                            raw_text=str(obj.get("text") or ""),
                            source_path=store_dir,
                        )
                    )
            self.records = records
            return bool(self.records)
        except Exception as exc:
            print(f"[WARN] 无法加载已有 vector_db 索引，将尝试重建：{exc}")
            return False

    def _load_saved_index(self, index_dir: str) -> bool:
        index_path = os.path.join(index_dir, "index.faiss")
        meta_path = os.path.join(index_dir, "meta.jsonl")
        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False
        try:
            import faiss

            self.index = faiss.read_index(index_path)
            self.records = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    self.records.append(CaseRecord(**obj))
            return bool(self.records)
        except Exception as exc:
            print(f"[WARN] 无法加载 storage 索引，将尝试重建：{exc}")
            return False

    def _load_training_records(self) -> List[CaseRecord]:
        if not os.path.isdir(self.train_dir):
            print(f"[WARN] 训练集目录不存在：{self.train_dir}")
            print(DATA_PREP_HINT)
            return []

        paths = sorted(
            [
                os.path.join(self.train_dir, name)
                for name in os.listdir(self.train_dir)
                if name.endswith(".json") and os.path.isfile(os.path.join(self.train_dir, name))
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

    def build_index(self) -> None:
        self.records = self._load_training_records()
        if not self.records:
            self.index = None
            return
        try:
            import faiss
        except Exception as exc:
            print(f"[WARN] faiss 不可用，无法构建检索索引：{exc}")
            self.index = None
            return

        texts = [r.raw_text for r in self.records]
        vectors = self.embedding_engine.embed_texts(texts).astype("float32")
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

            os.makedirs(self.index_dir, exist_ok=True)
            faiss.write_index(self.index, os.path.join(self.index_dir, "index.faiss"))
            with open(os.path.join(self.index_dir, "meta.jsonl"), "w", encoding="utf-8") as f:
                for record in self.records:
                    f.write(json.dumps(record.__dict__, ensure_ascii=False) + "\n")
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
            results = []
            for score, idx in zip(scores[0], indices[0]):
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
        except Exception as exc:
            print(f"[WARN] FAISS 检索失败：{exc}")
            return []


def retrieve_similar_cases(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    return FaissCaseRetriever(top_k=top_k).retrieve_similar_cases(query_text, top_k=top_k)

