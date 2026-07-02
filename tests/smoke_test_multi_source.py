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

from engines.retrieval.multi_source_retriever import MultiSourceRetriever


def main() -> int:
    vector_db = os.path.join(ROOT, "vector_db")
    if not os.path.isdir(vector_db):
        print(f"[SKIP] vector_db 不存在：{vector_db}")
        print("请先运行：python scripts\\build_vector_stores.py --sources ddxplus_cases ddxplus_kg --local")
        return 0
    retriever = MultiSourceRetriever(base_dir=vector_db, force_local=True)
    if not retriever.sources:
        print("[SKIP] 没有可用的多源向量库。")
        return 0
    results = retriever.retrieve(
        "fever cough sore throat and night sweats",
        sources=["all"],
        top_k=4,
        top_k_per_source=2,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2)[:4000])
    if not results:
        print("[WARN] 多源向量检索没有返回结果。")
        return 0
    if not any(row.get("source") for row in results):
        print("[ERROR] multi-source results missing source field")
        return 1
    print("[OK] smoke_test_multi_source passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

