# -*- coding: utf-8 -*-
"""Chinese Gradio frontend aligned with the course project documents."""

from __future__ import annotations

import time
import os
import sys
from typing import Any, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from frontend.components import chat_tab, history_tab, result_tab, upload_tab

API_BASE = os.getenv("MEDRAG_API_BASE", "http://127.0.0.1:8000/api/v1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")


def _patch_starlette_template_response() -> None:
    """Allow older Gradio releases to run with newer Starlette templates."""
    try:
        import inspect

        from starlette.templating import Jinja2Templates
    except Exception:
        return

    original = Jinja2Templates.TemplateResponse
    if getattr(original, "_medrag_compat", False):
        return

    parameters = list(inspect.signature(original).parameters)
    if len(parameters) < 3 or parameters[1:3] != ["request", "name"]:
        return

    def template_response_compat(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", None)
            context = context or {}
            request = context.get("request")
            if request is not None:
                return original(self, request, name, context, *args[2:], **kwargs)
        return original(self, *args, **kwargs)

    template_response_compat._medrag_compat = True
    Jinja2Templates.TemplateResponse = template_response_compat


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _requests():
    import requests

    return requests


def diagnose_sync(text: str, top_k: int = 5) -> str:
    if not str(text or "").strip():
        return "请输入病例描述或体检报告文本。"
    resp = _requests().post(
        f"{API_BASE}/diagnosis/text/sync",
        json={"text": text, "top_k": int(top_k), "use_multi_agent": True, "vector_sources": ["all"]},
        timeout=180,
    )
    data: Dict[str, Any] = resp.json()
    report = data.get("report", {})
    return report.get("summary_markdown") or str(data)


def diagnose_async(text: str, top_k: int = 5) -> tuple[str, str]:
    if not str(text or "").strip():
        return "", "请输入文本。"
    resp = _requests().post(
        f"{API_BASE}/diagnosis/text",
        json={"text": text, "top_k": int(top_k), "use_multi_agent": True, "vector_sources": ["all"]},
        timeout=30,
    )
    data = resp.json()
    task_id = data["task_id"]
    for _ in range(120):
        status = _requests().get(f"{API_BASE}/tasks/{task_id}", timeout=10).json()
        if status.get("status") == "done":
            report = status.get("result") or {}
            return task_id, report.get("summary_markdown") or str(report)
        if status.get("status") == "failed":
            return task_id, status.get("error", "任务失败")
        time.sleep(1)
    return task_id, "任务超时，请稍后到历史记录页查看。"


def build_app():
    _patch_starlette_template_response()
    try:
        import gradio as gr
    except Exception as exc:
        return {"status": "gradio_unavailable", "error": str(exc)}

    with gr.Blocks(title="多模态医疗报告智能解读与多 Agent 辅助诊断系统") as demo:
        gr.Markdown("# 多模态医疗报告智能解读与多 Agent 辅助诊断系统")

        with gr.Tab("文本诊断"):
            text_input = gr.Textbox(
                label="病例或体检报告文本",
                lines=10,
                placeholder="示例：患者男，56 岁，血压 150/95 mmHg，LDL-C 4.2 mmol/L，GLU 7.1 mmol/L，ALT 68 U/L。",
            )
            top_k = gr.Slider(label="Top-K", minimum=1, maximum=20, value=5, step=1)
            btn = gr.Button("开始诊断")
            task_id_box = gr.Textbox(label="Task ID")
            output = gr.Markdown(label="诊断报告")
            btn.click(diagnose_async, inputs=[text_input, top_k], outputs=[task_id_box, output], show_api=False)

        with gr.Tab("同步调试"):
            text_input_sync = gr.Textbox(label="调试文本", lines=8)
            top_k_sync = gr.Slider(label="Top-K", minimum=1, maximum=20, value=5, step=1)
            btn_sync = gr.Button("同步运行")
            output_sync = gr.Markdown()
            btn_sync.click(diagnose_sync, inputs=[text_input_sync, top_k_sync], outputs=[output_sync], show_api=False)

        with gr.Tab("上传报告"):
            upload_tab.render(gr, _requests(), API_BASE)

        with gr.Tab("报告结果"):
            result_tab.render(gr, _requests(), API_BASE)

        with gr.Tab("智能追问"):
            chat_tab.render(gr, _requests(), API_BASE)

        with gr.Tab("历史记录"):
            history_tab.render(gr, _requests(), API_BASE)

    return demo


if __name__ == "__main__":
    app = build_app()
    if hasattr(app, "launch"):
        app.launch(
            server_name=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
            server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
            share=_env_bool("GRADIO_SHARE", False),
        )
    else:
        print(app)
