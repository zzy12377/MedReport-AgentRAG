# -*- coding: utf-8 -*-
"""
scripts/retrieve_multi_vector.py

多向量库联合检索工具。

示例：
    python scripts/retrieve_multi_vector.py --query "70-year-old with cough, night sweats, chest pain" --top-k 10 --local
    python scripts/retrieve_multi_vector.py --query "fever and rash" --sources ddxplus_cases pmc_patients --top-k 10
    python scripts/retrieve_multi_vector.py --query "shortness of breath" --sources all --top-k-per-source 3 --final-top-k 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Windows CMD 默认 GBK 编码，强制用 UTF-8 避免中文输出报错
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vector_store.registry import MultiVectorRetriever
from vector_store.utils import create_embedding_fn, create_query_embedding_fn


def _print_results(results: list) -> None:
    """格式化打印检索结果。"""
    if not results:
        print("[INFO] 无检索结果。")
        return

    # 计算列宽
    rank_w = 5
    score_w = 10
    source_w = min(max(len(r.get("source", "")) for r in results), 22)
    diag_w = min(max(len(r.get("diagnosis", "")) for r in results), 25)

    header = (
        f"{'Rank':<{rank_w}} {'Score':<{score_w}} {'Source':<{source_w}} "
        f"{'Diagnosis':<{diag_w}} Title / Text"
    )
    print("\n" + header)
    print("-" * len(header))

    for i, r in enumerate(results, 1):
        score = r.get("score", 0.0)
        source = r.get("source", "?")
        diagnosis = r.get("diagnosis", "")
        title = r.get("title", "")[:50]
        text = r.get("text", "")[:120].replace("\n", " ")

        print(
            f"{i:<{rank_w}} {score:<{score_w}.4f} {source:<{source_w}} "
            f"{diagnosis:<{diag_w}} {title} | {text}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="多向量库联合检索",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="查询文本。",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="要检索的向量库名称。省略则检索所有可用库。",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="最终返回结果数（默认 10）。",
    )
    parser.add_argument(
        "--top-k-per-source",
        type=int,
        default=5,
        help="每个库检索的结果数（默认 5）。",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="仅使用本地 embedding。",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="保存结果到 JSON 文件。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每条结果的完整 text 字段。",
    )

    args = parser.parse_args()

    if args.sources and any(str(source).lower() == "all" for source in args.sources):
        args.sources = None

    # 创建 embedding 函数
    print("[INFO] Loading embedding model...")
    try:
        embedding_fn = create_embedding_fn(force_local=args.local)
        query_embedding_fn = create_query_embedding_fn(embedding_fn)
    except Exception as e:
        print(f"[ERROR] 创建 embedding 函数失败：{e}")
        return 1

    # 创建检索器
    retriever = MultiVectorRetriever(base_dir="./vector_db")

    if not retriever.sources:
        print("[ERROR] 未找到任何向量库。请先运行 build_vector_stores.py 构建。")
        return 1

    # 检索
    print(f"\n[INFO] Query: {args.query}")
    print(f"[INFO] Sources: {args.sources or 'all'}")
    print(f"[INFO] Top-K per source: {args.top_k_per_source}, Final Top-K: {args.top_k}")

    results = retriever.search(
        query=args.query,
        embedding_fn=query_embedding_fn,
        sources=args.sources,
        top_k_per_source=args.top_k_per_source,
        final_top_k=args.top_k,
    )

    # 打印结果
    _print_results(results)

    if args.verbose and results:
        print("\n" + "=" * 70)
        print("[DETAIL] 完整字段：")
        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            for k, v in r.items():
                if k != "metadata":
                    val = str(v)[:300]
                    print(f"  {k}: {val}")
            # metadata 单独一行
            if "metadata" in r:
                print(f"  metadata: {json.dumps(r['metadata'], ensure_ascii=False)[:200]}")

    # 可选保存
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
