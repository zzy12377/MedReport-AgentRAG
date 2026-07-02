# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from engines.kg.kg_extractor import DEFAULT_KG_PATH, DDXPlusKGRetriever


def main() -> int:
    kg_path = os.path.join(ROOT, DEFAULT_KG_PATH)
    if not os.path.exists(kg_path):
        print(f"[SKIP] KG 文件不存在：{kg_path}")
        return 0
    retriever = DDXPlusKGRetriever(kg_path=kg_path)
    evidence = retriever.retrieve(
        query_text="fever cough sore throat and night sweats",
        entities=[],
        top_k=5,
    )
    print(json.dumps(evidence, ensure_ascii=False, indent=2)[:4000])
    if not evidence:
        print("[WARN] KG 检索没有返回证据。")
        return 0
    required = {"head", "relation", "tail", "score", "relation_category", "retrieval_method", "neighbors"}
    missing = required - set(evidence[0])
    if missing:
        print(f"[ERROR] KG evidence missing fields: {sorted(missing)}")
        return 1
    print("[OK] smoke_test_kg_hybrid passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

