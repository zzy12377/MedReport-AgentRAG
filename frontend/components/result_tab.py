from __future__ import annotations

import json


def render(gr, requests, api_base: str):
    task_id = gr.Textbox(label="Task ID")
    load_btn = gr.Button("加载报告")
    report_md = gr.Markdown(label="诊断报告")
    raw_json = gr.Textbox(label="结构化结果 JSON", lines=18, interactive=False)

    def load_report(tid: str):
        tid = str(tid or "").strip()
        if not tid:
            return "请输入 Task ID。", ""
        resp = requests.get(f"{api_base}/diagnosis/{tid}/report", timeout=30)
        if resp.status_code != 200:
            return f"未找到报告：{resp.text}", ""
        report = resp.json()
        raw_text = json.dumps(report, ensure_ascii=False, indent=2)
        return report.get("summary_markdown") or raw_text, raw_text

    load_btn.click(load_report, inputs=[task_id], outputs=[report_md, raw_json], show_api=False)
