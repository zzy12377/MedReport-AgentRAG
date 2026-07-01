# -*- coding: utf-8 -*-
"""Lightweight KG evidence extraction skeleton."""

from __future__ import annotations

from typing import Dict, Iterable, List


def extract_kg_evidence(entities: Iterable[Dict[str, object]], top_k: int = 10) -> List[Dict[str, object]]:
    evidence = []
    for entity in entities:
        name = str(entity.get("name", ""))
        if not entity.get("is_abnormal"):
            continue
        evidence.append(
            {
                "head": name,
                "relation": "may_indicate",
                "tail": "requires specialist review",
                "source": "rule_stub",
            }
        )
        if len(evidence) >= top_k:
            break
    return evidence

