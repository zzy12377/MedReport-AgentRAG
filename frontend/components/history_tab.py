from __future__ import annotations


def render(gr, requests, api_base: str):
    refresh_btn = gr.Button("刷新历史")
    history_table = gr.Dataframe(headers=["task_id", "overall_risk", "diagnoses"], datatype=["str", "str", "str"])

    def load_history():
        resp = requests.get(f"{api_base}/history", timeout=30)
        data = resp.json()
        rows = []
        for item in data.get("items", []):
            report = item.get("report") or {}
            rows.append(
                [
                    item.get("task_id") or report.get("task_id") or "",
                    report.get("overall_risk", ""),
                    ", ".join(report.get("possible_diagnoses") or []),
                ]
            )
        return rows

    refresh_btn.click(load_history, outputs=[history_table])
