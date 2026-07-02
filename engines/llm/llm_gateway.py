# -*- coding: utf-8 -*-
"""Unified LLM gateway with a deterministic mock fallback."""

from __future__ import annotations

import re
import os
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
        labels.append("cardiovascular risk")
    if any(k in text for k in ["alt", "ast", "ggt", "tbil", "肝"]):
        labels.append("liver function abnormality")
    if any(k in text for k in ["glu", "hba1c", "glucose", "糖"]):
        labels.append("endocrine/metabolic risk")
    if any(k in text for k in ["cough", "fever", "night sweats", "咳", "发热"]):
        labels.append("respiratory infection differential")
    if not labels:
        labels.append("general medical risk")
    return (
        f"Mock {mode} diagnosis: {', '.join(labels)}.\n"
        "This mock output is used because no available LLM call was completed."
    )


class LLMGateway:
    def __init__(self, model: Optional[str] = None, mock: bool = False):
        self.model = model
        self.mock = mock

    def generate(self, prompt: str, system_prompt: Optional[str] = None, mode: str = "B0") -> str:
        if self.mock:
            return mock_generate(prompt, mode=mode)

        try:
            import openai

            api_key = os.getenv("LLM_API_KEY", "")
            base_url = os.getenv("LLM_BASE_URL", "")
            chat_model = os.getenv("LLM_CHAT_MODEL", "")
            if not api_key:
                try:
                    from authentication import api_key as auth_api_key
                    from authentication import base_url as auth_base_url
                    from authentication import chat_model as auth_chat_model

                    api_key = auth_api_key
                    base_url = auth_base_url
                    chat_model = auth_chat_model
                except Exception:
                    api_key = ""
                    base_url = ""
                    chat_model = ""
            if not api_key or "your_" in str(api_key).lower():
                return mock_generate(prompt, mode=mode)
            model_to_use = self.model or chat_model
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.2,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            return mock_generate(f"{prompt}\nLLM error: {exc}", mode=mode)


def extract_prediction(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return ""
    match = re.search(r"(?:Diagnosis|诊断|prediction)\s*[:：]\s*([^.;。]+)", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned[:120]
