# -*- coding: utf-8 -*-
"""Minimal chat route placeholder."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatRequest(BaseModel):
    task_id: str | None = None
    message: str


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    return {
        "task_id": req.task_id,
        "reply": "Chat follow-up is reserved for the next phase. Please use the diagnosis report as current context.",
    }
