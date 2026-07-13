# -*- coding: utf-8 -*-
"""Unified LLM gateway with a deterministic mock fallback.

The gateway talks to OpenAI-compatible chat-completions APIs. Ollama exposes
that API at http://127.0.0.1:11434/v1, so the same path can be used for local
Qwen models without importing provider-specific authentication modules.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def mock_generate(prompt: str, mode: str = "B0") -> str:
    text = str(prompt or "").lower()
    labels = []
    if any(k in text for k in ["ldl", "hdl", "tc", "tg", "blood pressure", "血压", "收缩压"]):
        labels.append("心血管代谢风险")
    if any(k in text for k in ["alt", "ast", "ggt", "tbil", "肝"]):
        labels.append("肝功能指标异常")
    if any(k in text for k in ["glu", "hba1c", "glucose", "糖"]):
        labels.append("血糖或内分泌代谢风险")
    if any(k in text for k in ["cough", "fever", "night sweats", "咳", "发热"]):
        labels.append("呼吸道感染相关鉴别")
    if not labels:
        labels.append("一般医学风险")
    return (
        f"模拟 {mode} 诊断摘要：{ '、'.join(labels) }。\n"
        "当前使用本地模拟输出，因为没有完成可用的大模型调用。"
    )


class LLMGateway:
    def __init__(self, model: Optional[str] = None, mock: bool = False):
        self.model = model
        self.mock = mock

    def generate(self, prompt: str, system_prompt: Optional[str] = None, mode: str = "B0") -> str:
        if self.mock:
            return mock_generate(prompt, mode=mode)

        try:
            api_key = os.getenv("LLM_API_KEY", "")
            base_url = _normalize_base_url(os.getenv("LLM_BASE_URL", ""))
            chat_model = os.getenv("LLM_CHAT_MODEL", "")
            if not api_key:
                try:
                    from authentication import api_key as auth_api_key
                    from authentication import base_url as auth_base_url
                    from authentication import chat_model as auth_chat_model

                    api_key = auth_api_key
                    base_url = _normalize_base_url(auth_base_url)
                    chat_model = auth_chat_model
                except Exception:
                    api_key = ""
                    base_url = ""
                    chat_model = ""
            if _is_local_llm_url(base_url) and (not api_key or "your_" in str(api_key).lower()):
                api_key = "ollama"
            if not api_key or "your_" in str(api_key).lower():
                return mock_generate(prompt, mode=mode)
            model_to_use = self.model or chat_model
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            try:
                import openai

                client = openai.OpenAI(api_key=api_key, base_url=base_url)
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=messages,
                    temperature=0.2,
                )
                return response.choices[0].message.content or ""
            except ImportError:
                return _generate_openai_compatible_http(
                    base_url=base_url,
                    api_key=api_key,
                    model=model_to_use,
                    messages=messages,
                )
        except Exception as exc:
            if _env_bool("LLM_STRICT_LOCAL", False) and _is_local_llm_url(os.getenv("LLM_BASE_URL", "")):
                raise
            return mock_generate(f"{prompt}\nLLM error: {exc}", mode=mode)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_base_url(base_url: object) -> str:
    value = str(base_url or "").strip().rstrip("/")
    if not value:
        return ""
    if _is_ollama_host(value) and not value.endswith("/v1"):
        return value + "/v1"
    return value


def _is_ollama_host(base_url: str) -> bool:
    value = str(base_url or "").lower()
    return "127.0.0.1:11434" in value or "localhost:11434" in value


def _is_local_llm_url(base_url: str) -> bool:
    value = str(base_url or "").lower()
    return (
        "127.0.0.1" in value
        or "localhost" in value
        or value.startswith("http://[::1]")
    )


def _generate_openai_compatible_http(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    if not base_url:
        raise ValueError("LLM_BASE_URL is empty")
    if not model:
        raise ValueError("LLM_CHAT_MODEL is empty")
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "stream": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key or 'ollama'}",
        },
        method="POST",
    )
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    return str((message or {}).get("content") or "")


def extract_prediction(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    match = re.search(r"(?:Diagnosis|诊断|prediction)\s*[:：]\s*([^.;。]+)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned[:120]
