from __future__ import annotations


def render(gr, requests, api_base: str):
    task_id = gr.Textbox(label="Task ID")
    message = gr.Textbox(label="追问", lines=3, placeholder="例如：这些指标最需要复查什么？")
    send_btn = gr.Button("发送")
    reply = gr.Markdown()

    def ask(tid: str, msg: str):
        if not str(msg or "").strip():
            return "请输入追问。"
        resp = requests.post(f"{api_base}/chat", json={"task_id": tid or None, "message": msg}, timeout=90)
        data = resp.json()
        return data.get("reply") or str(data)

    send_btn.click(ask, inputs=[task_id, message], outputs=[reply])
