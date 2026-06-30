# -*- coding: utf-8 -*-
"""
scripts/build_vector_stores.py

构建多个独立向量库。

示例：
    # 只构建 ddxplus_cases，100 条，强制覆盖，纯本地
    python scripts/build_vector_stores.py --sources ddxplus_cases --max-per-source 100 --force --local

    # 构建所有可用数据源
    python scripts/build_vector_stores.py --sources all --max-per-source 5000 --batch-size 8

    # 构建指定几个源
    python scripts/build_vector_stores.py --sources ddxplus_cases ddxplus_kg pmc_patients --max-per-source 1000
"""

from __future__ import annotations

import argparse
import os
import sys

# Windows CMD 默认 GBK 编码，强制用 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vector_store.adapters import ADAPTER_MAP
from vector_store.builder import build_faiss_store
from vector_store.utils import create_embedding_fn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="构建 MedRAG 多向量库",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python scripts/build_vector_stores.py --sources ddxplus_cases --max-per-source 100 --force --local
  python scripts/build_vector_stores.py --sources all --max-per-source 5000
  python scripts/build_vector_stores.py --sources ddxplus_cases ddxplus_kg pmc_patients
        """,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="要构建的向量库名称，如 ddxplus_cases ddxplus_kg pmc_patients。特殊值 'all' 表示所有可用源。",
    )
    parser.add_argument(
        "--max-per-source",
        type=int,
        default=5000,
        help="每个源最多读取的记录数（默认 5000）。设为 0 表示不限制。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="每批传给 embedding 函数的文本数（默认 8）。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已有向量库。",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="仅使用本地 sentence-transformers embedding，不调用远程 API。",
    )
    parser.add_argument(
        "--output-base",
        default="./vector_db",
        help="向量库根目录（默认 ./vector_db）。",
    )

    args = parser.parse_args()

    # 解析 sources
    if "all" in args.sources:
        all_sources = list(ADAPTER_MAP.keys())
        print(f"[INFO] 将构建所有 {len(all_sources)} 个源：{all_sources}")
        sources = all_sources
    else:
        # 验证 source 名称
        invalid = [s for s in args.sources if s not in ADAPTER_MAP]
        if invalid:
            print(f"[ERROR] 未知数据源：{invalid}")
            print(f"[INFO] 可用源：{list(ADAPTER_MAP.keys())}")
            return 1
        sources = args.sources

    # 创建 embedding 函数
    try:
        embedding_fn = create_embedding_fn(force_local=args.local)
    except ImportError as e:
        print(f"[ERROR] 创建 embedding 函数失败：{e}")
        print("[INFO] 请确认已安装所有依赖：pip install -r requirements.txt")
        return 1

    max_per = None if args.max_per_source == 0 else args.max_per_source

    # 构建每个向量库
    built_count = 0
    for source_name in sources:
        print("\n" + "=" * 70)
        print(f"[INFO] Building store: {source_name}")
        print("=" * 70)

        adapter_fn = ADAPTER_MAP[source_name]

        try:
            records = adapter_fn(max_rows=max_per)
        except TypeError:
            # 部分适配器不支持 max_rows 参数（如 multicare, synthea）
            records = adapter_fn()

        if not records:
            print(f"[WARN] {source_name}: 无记录，跳过构建。")
            continue

        output_dir = os.path.join(args.output_base, source_name)

        try:
            build_faiss_store(
                records=records,
                output_dir=output_dir,
                embedding_fn=embedding_fn,
                batch_size=args.batch_size,
                force=args.force,
            )
            built_count += 1
        except Exception as e:
            print(f"[ERROR] 构建 {source_name} 失败：{e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"[DONE] 构建完成。成功构建 {built_count}/{len(sources)} 个向量库。")
    print(f"[INFO] 向量库根目录：{os.path.abspath(args.output_base)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
