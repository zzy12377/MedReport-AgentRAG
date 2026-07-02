# -*- coding: utf-8 -*-
"""DDXPlus KG evidence retrieval.

This phase keeps KG retrieval lightweight and deterministic: it loads all KG
triples from the Excel file, scores them by token overlap against the case text
and extracted entities, and returns the top evidence triples.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence


DEFAULT_KG_PATH = "./dataset/knowledge graph of DDXPlus.xlsx"


def _norm(text: object) -> str:
    value = str(text or "").lower()
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff%./ ]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _tokens(text: object) -> set[str]:
    return {t for t in _norm(text).split() if len(t) >= 2}


@lru_cache(maxsize=2)
def load_kg_triples(kg_path: str = DEFAULT_KG_PATH) -> List[Dict[str, str]]:
    kg_path = os.path.normpath(kg_path)
    if not os.path.exists(kg_path):
        print(f"[WARN] KG 文件不存在：{kg_path}")
        return []
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
            }
        )
    print(f"[INFO] Loaded KG triples: {len(triples)} from {kg_path}")
    return triples


class DDXPlusKGRetriever:
    def __init__(self, kg_path: str = DEFAULT_KG_PATH):
        self.kg_path = kg_path
        self.triples = load_kg_triples(kg_path)

    def retrieve(
        self,
        query_text: str = "",
        entities: Sequence[Dict[str, object]] | None = None,
        top_k: int = 10,
    ) -> List[Dict[str, object]]:
        if not self.triples:
            return []
        entity_terms = []
        for entity in entities or []:
            entity_terms.append(str(entity.get("name", "")))
            if entity.get("is_abnormal"):
                entity_terms.append(f"{entity.get('name')} abnormal")
        query = " ".join([query_text or ""] + entity_terms)
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        scored = []
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
            item = dict(triple)
            item["score"] = float(score)
            scored.append(item)

        scored.sort(key=lambda row: row.get("score", 0.0), reverse=True)
        return scored[: max(0, int(top_k))]


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

