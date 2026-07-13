# -*- coding: utf-8 -*-
"""
vector_store/utils.py

创建 embedding 可调用对象，封装 embedding_backend.py 的远程/本地双后端逻辑。

create_embedding_fn(force_local=False) -> Callable[[List[str]], np.ndarray]
"""

from __future__ import annotations

from typing import Callable, List

import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, **_: object):
        return iterable

from embedding_backend import (
    get_embedding_state,
    set_embedding_state,
    local_sentence_transformer_embeddings,
    safe_model_name,
)


def create_embedding_fn(force_local: bool = False) -> Callable[[List[str]], np.ndarray]:
    """
    创建一个 embedding 函数，供 builder 和 retriever 使用。

    Args:
        force_local: True 则直接使用本地 sentence-transformers，不尝试远程 API。

    Returns:
        fn(texts: List[str]) -> np.ndarray，形状 (N, dim)，float32 且 L2 归一化。
    """
    if force_local:
        def _local_fn(texts: List[str]) -> np.ndarray:
            return local_sentence_transformer_embeddings(
                texts, reason="forced local mode (--local flag)"
            )
        print("[INFO] Embedding mode: local sentence-transformers only")
        return _local_fn

    # 尝试远程 SiliconFlow API，失败则回退本地
    try:
        from authentication import api_key, base_url, embedding_model
        import openai
        try:
            import faiss  # type: ignore
        except Exception:
            faiss = None

        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        def _remote_with_fallback(texts: List[str]) -> np.ndarray:
            state = get_embedding_state()
            if state.get("backend") == "local":
                return local_sentence_transformer_embeddings(
                    texts,
                    reason=f"already using local model: {state.get('model')}",
                )

            try:
                embeddings = []
                for text in tqdm(texts, desc="Remote embedding"):
                    response = client.embeddings.create(
                        input=text,
                        model=embedding_model,
                    )
                    embeddings.append(response.data[0].embedding)

                vectors = np.asarray(embeddings, dtype="float32")
                if faiss is not None:
                    faiss.normalize_L2(vectors)
                else:
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    vectors = vectors / np.maximum(norms, 1e-12)
                set_embedding_state(
                    backend="remote",
                    model=embedding_model,
                    dim=int(vectors.shape[1]),
                )
                return vectors

            except Exception as e:
                print(f"[WARN] Remote embedding API failed: {e}")
                print("[WARN] Falling back to local sentence-transformers...")
                return local_sentence_transformer_embeddings(
                    texts,
                    reason=f"remote API error: {e}",
                )

        print(f"[INFO] Embedding mode: remote ({embedding_model}) + local fallback")
        return _remote_with_fallback

    except ImportError:
        print("[WARN] Cannot import authentication or openai, using local embedding only")
        def _fallback_fn(texts: List[str]) -> np.ndarray:
            return local_sentence_transformer_embeddings(
                texts, reason="remote not configured"
            )
        return _fallback_fn


def create_query_embedding_fn(embedding_fn: Callable[[List[str]], np.ndarray]) -> Callable[[str], np.ndarray]:
    """
    将批量 embedding 函数包装成单条查询 embedding 函数。
    retriever 需要 fn(query: str) -> np.ndarray 签名。
    """
    def _query_fn(query: str) -> np.ndarray:
        vecs = embedding_fn([query])
        return vecs[0]
    return _query_fn
