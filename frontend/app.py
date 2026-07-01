# -*- coding: utf-8 -*-
"""Gradio frontend placeholder.

The module imports without Gradio. Install Gradio in a later phase to launch
the UI.
"""

from __future__ import annotations


def build_app():
    try:
        import gradio as gr
    except Exception as exc:
        return {"status": "gradio_unavailable", "error": str(exc)}

    with gr.Blocks(css="./frontend/styles/custom.css") as demo:
        gr.Markdown("# 多模态医疗报告智能解读与多 Agent 辅助诊断系统")
        gr.Markdown("第一阶段：B0/B1 baseline 与模块化骨架。")
    return demo


if __name__ == "__main__":
    app = build_app()
    if hasattr(app, "launch"):
        app.launch()
    else:
        print(app)

