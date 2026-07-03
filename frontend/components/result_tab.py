from __future__ import annotations


def render(gr, requests, api_base: str):
    task_id = gr.Textbox(label="Task ID")
    load_btn = gr.Button("加载报告")
    report_md = gr.Markdown()
    raw_json = gr.JSON(label="结构化结果")

    def load_report(tid: str):
        tid = str(tid or "").strip()
        if not tid:
            return "请输入 Task ID。", {}
        resp = requests.get(f"{api_base}/diagnosis/{tid}/report", timeout=30)
        if resp.status_code != 200:
            return f"未找到报告：{resp.text}", {}
        report = resp.json()
        return report.get("summary_markdown") or str(report), report

    load_btn.click(load_report, inputs=[task_id], outputs=[report_md, raw_json])
