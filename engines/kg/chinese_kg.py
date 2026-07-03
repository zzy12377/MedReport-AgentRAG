# -*- coding: utf-8 -*-
"""Small Chinese physical-exam KG retriever.

This KG covers the demo domains in the course project: cardiovascular, liver,
endocrine/metabolic, blood routine and inflammation. It is rule-based and
dependency-free so the API can always import it.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List


DEFAULT_CHINESE_KG_PATH = "./resources/chinese_medical_kg.json"


def _direction(entity: Dict[str, Any]) -> str:
    value = entity.get("value")
    low = entity.get("ref_low")
    high = entity.get("ref_high")
    try:
        if low is not None and float(value) < float(low):
            return "low"
        if high is not None and float(value) > float(high):
            return "high"
    except Exception:
        pass
    return "abnormal" if entity.get("is_abnormal") else "normal"


@lru_cache(maxsize=4)
def load_chinese_kg(path: str = DEFAULT_CHINESE_KG_PATH) -> Dict[str, Any]:
    path = os.path.normpath(path)
    if not os.path.exists(path):
        return {"nodes": [], "edges": [], "node_map": {}, "out_edges": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = list(data.get("nodes") or [])
    edges = list(data.get("edges") or [])
    node_map = {str(node.get("id")): node for node in nodes}
    out_edges: Dict[str, List[Dict[str, Any]]] = {}
    for edge in edges:
        out_edges.setdefault(str(edge.get("source")), []).append(edge)
    data["node_map"] = node_map
    data["out_edges"] = out_edges
    return data


class ChineseMedicalKGRetriever:
    def __init__(self, kg_path: str = DEFAULT_CHINESE_KG_PATH):
        self.kg_path = kg_path
        self.graph = load_chinese_kg(kg_path)
        self.nodes = list(self.graph.get("nodes") or [])
        self.edges = list(self.graph.get("edges") or [])
        self.node_map: Dict[str, Dict[str, Any]] = dict(self.graph.get("node_map") or {})
        self.out_edges: Dict[str, List[Dict[str, Any]]] = dict(self.graph.get("out_edges") or {})

    def _feature_nodes_for(self, entity: Dict[str, Any]) -> List[Dict[str, Any]]:
        name = str(entity.get("name") or "")
        direction = _direction(entity)
        matches = []
        for node in self.nodes:
            if node.get("type") != "feature" or node.get("indicator") != name:
                continue
            node_direction = str(node.get("direction") or "")
            if direction in {"high", "low"} and node_direction and node_direction != direction:
                continue
            if direction == "normal":
                continue
            matches.append(node)
        return matches

    def _neighbors(self, node_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        rows = []
        for edge in self.out_edges.get(node_id, []):
            target = self.node_map.get(str(edge.get("target")), {})
            rows.append(
                {
                    "head": self.node_map.get(node_id, {}).get("name", node_id),
                    "relation": edge.get("relation"),
                    "tail": target.get("name", edge.get("target")),
                    "level": target.get("level"),
                    "type": target.get("type"),
                }
            )
            if len(rows) >= limit:
                break
        return rows

    def retrieve(self, entities: Iterable[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        seen = set()
        for entity in entities or []:
            for feature in self._feature_nodes_for(dict(entity)):
                for edge in self.out_edges.get(str(feature.get("id")), []):
                    target = self.node_map.get(str(edge.get("target")), {})
                    key = (feature.get("id"), edge.get("relation"), edge.get("target"))
                    if key in seen:
                        continue
                    seen.add(key)
                    abnormal_strength = 1.0 if entity.get("is_abnormal") else 0.3
                    item = {
                        "head": feature.get("name"),
                        "relation": edge.get("relation"),
                        "tail": target.get("name", edge.get("target")),
                        "text": f"{feature.get('name')} {edge.get('relation')} {target.get('name', edge.get('target'))}",
                        "source": "Chinese_Physical_Exam_KG",
                        "relation_category": "test_indicator",
                        "score": abnormal_strength * 10.0,
                        "retrieval_method": "indicator_rule",
                        "indicator": entity.get("name"),
                        "value": entity.get("value"),
                        "unit": entity.get("unit"),
                        "direction": _direction(dict(entity)),
                        "neighbors": self._neighbors(str(edge.get("target"))),
                    }
                    item["neighbor_count"] = len(item["neighbors"])
                    evidence.append(item)
        evidence.sort(key=lambda row: float(row.get("score", 0.0)), reverse=True)
        return evidence[: max(0, int(top_k))]
