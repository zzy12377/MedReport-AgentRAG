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

from engines.retrieval.faiss_retriever import DATA_PREP_HINT, FaissCaseRetriever


def main() -> int:
    train_dir = os.path.join(ROOT, "dataset", "df", "train")
    if not os.path.isdir(train_dir):
        print(f"[SKIP] 缺少训练集目录：{train_dir}")
        print(DATA_PREP_HINT)
        return 0
    retriever = FaissCaseRetriever(train_dir=train_dir, top_k=2, force_local=True)
    results = retriever.retrieve_similar_cases("fever cough sore throat", top_k=2)
    print(json.dumps(results, ensure_ascii=False, indent=2)[:3000])
    if not results:
        print("[WARN] 没有检索结果；请确认 FAISS 依赖和训练集数据。")
        return 0
    print("[OK] smoke_test_faiss passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
