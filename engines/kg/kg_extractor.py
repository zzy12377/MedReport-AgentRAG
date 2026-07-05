# -*- coding: utf-8 -*-
"""DDXPlus KG evidence retrieval.

The retriever loads the full KG Excel file and can optionally merge evidence
from the prebuilt ``vector_db/ddxplus_kg`` vector store. It remains a weak
dependency: if the vector store is missing, rule-based KG retrieval still works.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_KG_PATH = "./dataset/knowledge graph of DDXPlus.xlsx"
DEFAULT_VECTOR_BASE_DIR = "./vector_db"


def _norm(text: object) -> str:
    value = str(text or "").lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff%./ ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: object) -> set[str]:
    return {t for t in _norm(text).split() if len(t) >= 2}


def _triple_key(head: object, relation: object, tail: object) -> Tuple[str, str, str]:
    return (_norm(head), _norm(relation), _norm(tail))


def classify_relation(relation: object, head: object = "", tail: object = "") -> str:
    text = _norm(" ".join([str(relation or ""), str(head or ""), str(tail or "")]))
    if any(term in text for term in ["symptom", "symptomatology", "anamnesis", "cough", "pain", "fever"]):
        return "symptom"
    if any(term in text for term in ["risk", "factor", "exposure", "history", "travel", "smoking"]):
        return "risk_factor"
    if any(term in text for term in ["test", "lab", "blood", "glucose", "alt", "ast", "ldl", "hba1c"]):
        return "test_indicator"
    if any(term in text for term in ["treatment", "drug", "medication", "therapy"]):
        return "treatment"
    if any(term in text for term in ["complication", "causes", "associated"]):
        return "pathophysiology"
    return "general"


def _with_relation_metadata(triple: Dict[str, object]) -> Dict[str, object]:
    item = dict(triple)
    item["relation_category"] = classify_relation(item.get("relation"), item.get("head"), item.get("tail"))
    return item


@lru_cache(maxsize=2)
def load_kg_triples(kg_path: str = DEFAULT_KG_PATH) -> List[Dict[str, str]]:
    kg_path = os.path.normpath(kg_path)
    if not os.path.exists(kg_path):
        print(f"[WARN] KG 文件不存在：{kg_path}")
        return []
    if kg_path.lower().endswith(".jsonl"):
        triples = []
        try:
            import json

            with open(kg_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    subject = str(row.get("head") or row.get("subject") or "").strip()
                    relation = str(row.get("relation") or "").strip()
                    obj = str(row.get("tail") or row.get("object") or "").strip()
                    if not (subject and relation and obj):
                        continue
                    text = str(row.get("text") or f"{subject} {relation} {obj}")
                    triples.append(
                        {
                            "head": subject,
                            "relation": relation,
                            "tail": obj,
                            "text": text,
                            "source": str(row.get("source") or "DDXPlus_KG_JSONL"),
                            "relation_category": classify_relation(relation, subject, obj),
                        }
                    )
        except Exception as exc:
            print(f"[WARN] 无法读取 KG JSONL 文件：{exc}")
            return []
        print(f"[INFO] Loaded KG triples: {len(triples)} from {kg_path}")
        return triples
    try:
        import pandas as pd

        df = pd.read_excel(kg_path, usecols=["subject", "relation", "object"])
    except Exception as exc:
        print(f"[WARN] 无法读取 KG 文件：{exc}")
        return []

    triples = []
    for _, row in df.dropna(subset=["subject", "relation", "object"]).iterrows():
        subject = str(row["subject"]).strip()
        relation = str(row["relation"]).strip()
        obj = str(row["object"]).strip()
        text = f"{subject} {relation.replace('_', ' ')} {obj}"
        triples.append(
            {
                "head": subject,
                "relation": relation,
                "tail": obj,
                "text": text,
                "source": "DDXPlus_KG",
                "relation_category": classify_relation(relation, subject, obj),
            }
        )
    print(f"[INFO] Loaded KG triples: {len(triples)} from {kg_path}")
    return triples


class DDXPlusKGRetriever:
    def __init__(
        self,
        kg_path: str = DEFAULT_KG_PATH,
        vector_base_dir: str = DEFAULT_VECTOR_BASE_DIR,
        use_vector: bool = True,
        subgraph_limit: int = 3,
    ):
        self.kg_path = kg_path
        self.triples = load_kg_triples(kg_path)
        self.vector_base_dir = vector_base_dir
        self.use_vector = use_vector
        self.subgraph_limit = max(0, int(subgraph_limit))
        self._triple_lookup = {
            _triple_key(t.get("head"), t.get("relation"), t.get("tail")): t
            for t in self.triples
        }
        self._kg_vector_index = None
        self._kg_vector_records: List[Dict[str, object]] = []
        self._kg_query_embedding_fn = None
        if use_vector and self._has_kg_vector_store():
            try:
                from vector_store.retriever import load_store
                from vector_store.utils import create_embedding_fn, create_query_embedding_fn

                store_dir = os.path.join(os.path.normpath(vector_base_dir), "ddxplus_kg")
                self._kg_vector_index, self._kg_vector_records = load_store(store_dir)
                embedding_fn = create_embedding_fn(force_local=True)
                self._kg_query_embedding_fn = create_query_embedding_fn(embedding_fn)
            except Exception as exc:
                print(f"[WARN] KG vector retriever unavailable, using rule KG only: {exc}")

    def _has_kg_vector_store(self) -> bool:
        store_dir = os.path.join(os.path.normpath(self.vector_base_dir), "ddxplus_kg")
        return os.path.exists(os.path.join(store_dir, "index.faiss")) and os.path.exists(os.path.join(store_dir, "meta.jsonl"))

    def _query_text(self, query_text: str, entities: Sequence[Dict[str, object]] | None = None) -> str:
        entity_terms = []
        for entity in entities or []:
            entity_terms.append(str(entity.get("name", "")))
            if entity.get("is_abnormal"):
                entity_terms.append(f"{entity.get('name')} abnormal")
        return " ".join([query_text or ""] + entity_terms)

    def _rule_retrieve(self, query: str) -> List[Dict[str, object]]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        scored: List[Dict[str, object]] = []
        for triple in self.triples:
            triple_tokens = _tokens(triple.get("text", ""))
            overlap = len(query_tokens & triple_tokens)
            substring_bonus = 0
            triple_text = _norm(triple.get("text", ""))
            for token in query_tokens:
                if token in triple_text:
                    substring_bonus += 1
            score = overlap * 2 + substring_bonus
            if score <= 0:
                continue
            item = _with_relation_metadata(triple)
            item["score"] = float(score)
            item["retrieval_method"] = "rule"
            scored.append(item)

        scored.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        return scored

    def _vector_retrieve(self, query: str, top_k: int) -> List[Dict[str, object]]:
        if self._kg_vector_index is None or self._kg_query_embedding_fn is None or not self._kg_vector_records:
            return []
        try:
            import faiss
            import numpy as np

            query_vec = np.asarray(self._kg_query_embedding_fn(query), dtype="float32")
            if query_vec.ndim == 1:
                query_vec = query_vec.reshape(1, -1)
            if query_vec.shape[1] != self._kg_vector_index.d:
                print(
                    f"[WARN] KG query embedding dim {query_vec.shape[1]} != KG vector index dim {self._kg_vector_index.d}; "
                    "skip KG vector retrieval."
                )
                return []
            faiss.normalize_L2(query_vec)
            k = min(int(top_k), self._kg_vector_index.ntotal)
            scores, indices = self._kg_vector_index.search(query_vec, k)
        except Exception as exc:
            print(f"[WARN] KG vector retrieval failed, using rule KG only: {exc}")
            return []

        evidence = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._kg_vector_records):
                continue
            row = self._kg_vector_records[int(idx)]
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            head = metadata.get("subject") or metadata.get("head")
            relation = metadata.get("relation") or ""
            tail = metadata.get("object") or metadata.get("tail")
            if not head or not tail:
                parts = str(row.get("text", "")).split()
                if len(parts) >= 3:
                    head = parts[0]
                    relation = " ".join(parts[1:-1])
                    tail = parts[-1]
            key = _triple_key(head, relation, tail)
            base = self._triple_lookup.get(key, {})
            item = _with_relation_metadata(
                {
                    "head": base.get("head") or head,
                    "relation": base.get("relation") or relation,
                    "tail": base.get("tail") or tail,
                    "text": base.get("text") or row.get("text", ""),
                    "source": "DDXPlus_KG",
                }
            )
            item["score"] = round(float(score) * 5.0, 4)
            item["vector_similarity"] = float(score)
            item["retrieval_method"] = "vector"
            evidence.append(item)
        return evidence

    def _merge_evidence(self, rule_rows: List[Dict[str, object]], vector_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        merged: Dict[Tuple[str, str, str], Dict[str, object]] = {}
        for row in rule_rows + vector_rows:
            key = _triple_key(row.get("head"), row.get("relation"), row.get("tail"))
            if key not in merged:
                merged[key] = dict(row)
                continue
            existing = merged[key]
            existing["score"] = float(existing.get("score", 0.0)) + float(row.get("score", 0.0))
            methods = {str(existing.get("retrieval_method", "")), str(row.get("retrieval_method", ""))}
            existing["retrieval_method"] = "hybrid" if len(methods - {""}) > 1 else next(iter(methods - {""}), "rule")
            if row.get("vector_similarity") is not None:
                existing["vector_similarity"] = row.get("vector_similarity")
        return sorted(merged.values(), key=lambda row: float(row.get("score", 0.0)), reverse=True)

    def _neighbors_for(self, row: Dict[str, object]) -> List[Dict[str, object]]:
        if self.subgraph_limit <= 0:
            return []
        head_n = _norm(row.get("head"))
        tail_n = _norm(row.get("tail"))
        neighbors = []
        for triple in self.triples:
            if _triple_key(triple.get("head"), triple.get("relation"), triple.get("tail")) == _triple_key(
                row.get("head"), row.get("relation"), row.get("tail")
            ):
                continue
            if _norm(triple.get("head")) in {head_n, tail_n} or _norm(triple.get("tail")) in {head_n, tail_n}:
                neighbors.append(
                    {
                        "head": triple.get("head"),
                        "relation": triple.get("relation"),
                        "tail": triple.get("tail"),
                        "relation_category": triple.get("relation_category") or classify_relation(
                            triple.get("relation"), triple.get("head"), triple.get("tail")
                        ),
                    }
                )
            if len(neighbors) >= self.subgraph_limit:
                break
        return neighbors

    def retrieve(
        self,
        query_text: str = "",
        entities: Sequence[Dict[str, object]] | None = None,
        top_k: int = 10,
    ) -> List[Dict[str, object]]:
        if not self.triples:
            return []
        query = self._query_text(query_text, entities)
        rule_rows = self._rule_retrieve(query)
        vector_rows = self._vector_retrieve(query, top_k=max(int(top_k) * 2, 1))
        merged = self._merge_evidence(rule_rows, vector_rows)
        results = merged[: max(0, int(top_k))]
        for row in results:
            row["neighbors"] = self._neighbors_for(row)
            row["neighbor_count"] = len(row["neighbors"])
        return results


def extract_kg_evidence(
    entities: Iterable[Dict[str, object]],
    top_k: int = 10,
    query_text: str = "",
    kg_path: str = DEFAULT_KG_PATH,
) -> List[Dict[str, object]]:
    return DDXPlusKGRetriever(kg_path=kg_path).retrieve(
        query_text=query_text,
        entities=list(entities),
        top_k=top_k,
    )
