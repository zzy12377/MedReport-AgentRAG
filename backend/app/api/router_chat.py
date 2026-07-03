# -*- coding: utf-8 -*-
"""Follow-up chat route grounded in a generated report."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.app.config.settings import settings
from backend.app.core.llm.llm_gateway import LLMGateway
from backend.app.services.report_service import ReportService

router = APIRouter()
reports = ReportService()


class ChatRequest(BaseModel):
    task_id: str | None = None
    message: str


@router.post("/chat")
def chat(req: ChatRequest) -> dict:
    report = reports.load_report(req.task_id) if req.task_id else None
    if not report:
        return {
            "task_id": req.task_id,
            "reply": "没有找到对应报告。请先完成一次诊断，或在消息里提供更完整的病情/体检指标。",
            "used_report": False,
        }
    context = {
        "overall_risk": report.get("overall_risk"),
        "possible_diagnoses": report.get("possible_diagnoses"),
        "entities": report.get("entities"),
        "agent_opinions": report.get("agent_opinions"),
        "critique": report.get("critique"),
        "summary_markdown": report.get("summary_markdown"),
    }
    prompt = (
        "请基于以下课程演示用辅助诊断报告，回答用户追问。"
        "必须提醒不能替代医生诊断，回答要简洁、中文、可执行。\n\n"
        f"报告上下文：{context}\n\n用户追问：{req.message}"
    )
    reply = LLMGateway.get_instance(mock=settings.force_mock_llm).generate(
        prompt,
        system_prompt="你是谨慎的医疗报告辅助解释助手。",
        mode="CHAT",
    )
    return {
        "task_id": req.task_id,
        "reply": reply,
        "used_report": True,
    }
