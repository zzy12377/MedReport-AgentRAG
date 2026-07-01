# -*- coding: utf-8 -*-
"""
scripts/inspect_vector_store.py

检查单个向量库的配置和样本记录。

示例：
    python scripts/inspect_vector_store.py --source ddxplus_cases
    python scripts/inspect_vector_store.py --source pmc_patients --samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Windows CMD 默认 GBK 编码，强制用 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main() -> int:
    parser = argparse.ArgumentParser(description="检查向量库")
    parser.add_argument(
        "--source",
        required=True,
        help="要检查的向量库名称（vector_db/ 下的子目录名）。",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="显示多少条样本记录（默认 5）。",
    )

    args = parser.parse_args()

    store_dir = os.path.normpath(os.path.join("./vector_db", args.source))

    if not os.path.isdir(store_dir):
        print(f"[ERROR] 向量库目录不存在：{store_dir}")
        print("[INFO] 可用库：")
        base_dir = "./vector_db"
        if os.path.isdir(base_dir):
            for entry in sorted(os.listdir(base_dir)):
                entry_path = os.path.join(base_dir, entry)
                if os.path.isdir(entry_path) and os.path.exists(os.path.join(entry_path, "index.faiss")):
                    print(f"  - {entry}")
        return 1

    # 读取 config
    required_files = ["index.faiss", "meta.jsonl", "config.json"]
    missing_files = [
        name for name in required_files
        if not os.path.exists(os.path.join(store_dir, name))
    ]
    if missing_files:
        print(f"[ERROR] 向量库目录不完整：{store_dir}")
        print(f"[ERROR] 缺少文件：{', '.join(missing_files)}")
        print("[INFO] 请重新构建，例如：")
        print(f"  python scripts/build_vector_stores.py --sources {args.source} --force --local")
        return 1

    config_path = os.path.join(store_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        print("=" * 60)
        print("Config")
        print("=" * 60)
        for k, v in config.items():
            print(f"  {k:25s}: {v}")
    else:
        print("[WARN] 未找到 config.json")
        config = {}

    # 读取 meta 样本
    meta_path = os.path.join(store_dir, "meta.jsonl")
    if os.path.exists(meta_path):
        print("\n" + "=" * 60)
        print(f"Sample Records (first {args.samples})")
        print("=" * 60)

        with open(meta_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.samples:
                    break
                record = json.loads(line.strip())

                print(f"\n--- Record {i+1} ---")
                for key in ["id", "source", "title"]:
                    if key in record:
                        val = str(record[key])
                        if len(val) > 100:
                            val = val[:100] + "..."
                        print(f"  {key:15s}: {val}")

                if "diagnosis" in record and record["diagnosis"]:
                    print(f"  {'diagnosis':15s}: {record['diagnosis']}")

                text = str(record.get("text", ""))
                if len(text) > 200:
                    text = text[:200] + "..."
                print(f"  {'text':15s}: {text}")

                if "metadata" in record:
                    # 只打印前 5 个 metadata 键
                    meta_keys = list(record["metadata"].keys())[:5]
                    for mk in meta_keys:
                        val = str(record["metadata"][mk])
                        if len(val) > 80:
                            val = val[:80] + "..."
                        print(f"  meta.{mk:12s}: {val}")

    else:
        print("[WARN] 未找到 meta.jsonl")

    # 检查 index.faiss 是否存在
    index_path = os.path.join(store_dir, "index.faiss")
    if os.path.exists(index_path):
        import faiss
        index = faiss.read_index(index_path)
        print(f"\n[INFO] Index loaded: {index.ntotal} vectors, dim={index.d}")

        if config and config.get("num_records", 0) != index.ntotal:
            print(f"[WARN] config 记录数 ({config.get('num_records')}) 与 index 向量数 ({index.ntotal}) 不一致！")

    return 0


if __name__ == "__main__":
    sys.exit(main())
