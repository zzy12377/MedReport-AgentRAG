from __future__ import annotations

import time
from typing import Any, Dict


def render(gr, requests, api_base: str):
    file_input = gr.File(label="上传体检报告图片/PDF/文本", file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".csv"])
    upload_btn = gr.Button("上传并分析")
    task_id_box = gr.Textbox(label="Task ID", interactive=False)
    ocr_preview = gr.Textbox(label="OCR/文本预览", lines=6, interactive=False)
    report_md = gr.Markdown()

    def upload_and_wait(file_obj):
        if file_obj is None:
            return "", "", "请先选择文件。"
        path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
        with open(path, "rb") as f:
            resp = requests.post(f"{api_base}/reports/upload", files={"file": (path, f)}, timeout=180)
        data: Dict[str, Any] = resp.json()
        preview = data.get("ocr_text_preview") or data.get("message") or ""
        task_id = data.get("task_id") or ""
        if not task_id:
            return "", preview, f"上传完成，但 OCR 未成功：{data.get('message', '')}"
        for _ in range(120):
            status = requests.get(f"{api_base}/tasks/{task_id}", timeout=10).json()
            if status.get("status") == "done":
                report = status.get("result") or {}
                return task_id, preview, report.get("summary_markdown") or str(report)
            if status.get("status") == "failed":
                return task_id, preview, status.get("error", "任务失败")
            time.sleep(1)
        return task_id, preview, "任务仍在运行，请稍后到历史记录查看。"

    upload_btn.click(upload_and_wait, inputs=[file_input], outputs=[task_id_box, ocr_preview, report_md])
