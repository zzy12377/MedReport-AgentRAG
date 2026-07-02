# -*- coding: utf-8 -*-
"""Build the DDXPlus case FAISS index used by the runtime API."""

from __future__ import annotations

import argparse
import os

from backend.app.config.settings import settings
from engines.retrieval.faiss_retriever import FaissCaseRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DDXPlus case embeddings and FAISS index.")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    train_dir = args.train_dir or settings.preferred_train_dir()
    if not os.path.isdir(train_dir):
        print(f"[WARN] Train directory not found: {train_dir}")
        print("Next step: python scripts/prepare_ddxplus_for_medrag.py")
        return 0

    retriever = FaissCaseRetriever(train_dir=train_dir, top_k=args.top_k, force_local=True, force_rebuild=args.force)
    count = len(retriever.records)
    print(f"[DONE] Case index ready. records={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
