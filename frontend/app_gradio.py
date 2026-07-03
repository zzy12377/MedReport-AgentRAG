# -*- coding: utf-8 -*-
"""Gradio frontend aligned with the course project documents."""

from __future__ import annotations

import time
import os
from typing import Any, Dict

from frontend.components import chat_tab, history_tab, result_tab, upload_tab

API_BASE = os.getenv("MEDRAG_API_BASE", "http://127.0.0.1:8000/api/v1")


def _requests():
    import requests

    return requests


def diagnose_sync(text: str, top_k: int = 5) -> str:
    if not str(text or "").strip():
        return "Please enter a case description or medical report text."
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
        return "", "Please enter text."
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
            return task_id, status.get("error", "Task failed")
        time.sleep(1)
    return task_id, "Task timed out. Please check the history page later."


def build_app():
    try:
        import gradio as gr
    except Exception as exc:
        return {"status": "gradio_unavailable", "error": str(exc)}

    with gr.Blocks(title="MedReport AgentRAG") as demo:
        gr.Markdown("# MedReport AgentRAG")

        with gr.Tab("Text Diagnosis"):
            text_input = gr.Textbox(
                label="Case or report text",
                lines=10,
                placeholder="Example: Patient male, 56 years old, BP 150/95 mmHg, LDL-C 4.2 mmol/L, GLU 7.1 mmol/L, ALT 68 U/L.",
            )
            top_k = gr.Slider(label="Top-K", minimum=1, maximum=20, value=5, step=1)
            btn = gr.Button("Run Diagnosis")
            task_id_box = gr.Textbox(label="Task ID")
            output = gr.Markdown(label="Report")
            btn.click(diagnose_async, inputs=[text_input, top_k], outputs=[task_id_box, output])

        with gr.Tab("Sync Debug"):
            text_input_sync = gr.Textbox(label="Debug text", lines=8)
            top_k_sync = gr.Slider(label="Top-K", minimum=1, maximum=20, value=5, step=1)
            btn_sync = gr.Button("Run Sync")
            output_sync = gr.Markdown()
            btn_sync.click(diagnose_sync, inputs=[text_input_sync, top_k_sync], outputs=[output_sync])

        with gr.Tab("Upload"):
            upload_tab.render(gr, _requests(), API_BASE)

        with gr.Tab("Report"):
            result_tab.render(gr, _requests(), API_BASE)

        with gr.Tab("Chat"):
            chat_tab.render(gr, _requests(), API_BASE)

        with gr.Tab("History"):
            history_tab.render(gr, _requests(), API_BASE)

    return demo


if __name__ == "__main__":
    app = build_app()
    if hasattr(app, "launch"):
        app.launch(
            server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
            server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        )
    else:
        print(app)
