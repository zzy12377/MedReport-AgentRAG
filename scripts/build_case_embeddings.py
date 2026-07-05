# -*- coding: utf-8 -*-
"""Build the DDXPlus case FAISS index used by the runtime API."""

from __future__ import annotations

import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.config.settings import settings
from engines.retrieval.faiss_retriever import FaissCaseRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DDXPlus case embeddings and FAISS index.")
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--embeddings-path", default=None)
    parser.add_argument("--embedding-metadata-path", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    train_dir = args.train_dir or settings.preferred_train_dir()
    if not os.path.isdir(train_dir):
        print(f"[WARN] Train directory not found: {train_dir}")
        print("Next step: python scripts/prepare_ddxplus_for_medrag.py")
        return 0

    is_zh = settings.use_zh_data or "data_zh" in os.path.normpath(train_dir).lower()
    retriever = FaissCaseRetriever(
        train_dir=train_dir,
        index_path=args.index_path or (settings.case_zh_index_path if is_zh else settings.case_index_path),
        metadata_path=args.metadata_path or (settings.case_zh_index_metadata_path if is_zh else settings.case_index_metadata_path),
        embeddings_path=args.embeddings_path or (settings.case_zh_embeddings_path if is_zh else settings.case_embeddings_path),
        embedding_metadata_path=args.embedding_metadata_path
        or (settings.case_zh_embedding_metadata_path if is_zh else settings.case_embedding_metadata_path),
        top_k=args.top_k,
        force_local=True,
        force_rebuild=args.force,
    )
    count = len(retriever.records)
    if retriever.index is None:
        print(f"[WARN] Case records loaded but FAISS index was not built. records={count}")
        print("Next step: run this command inside the conda medrag environment with faiss installed.")
        return 0
    print(f"[DONE] Case index ready. records={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
