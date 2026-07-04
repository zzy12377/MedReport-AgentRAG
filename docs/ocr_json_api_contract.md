# OCR JSON Diagnosis API Contract

This document defines the frontend/backend contract for OCR-recognized report JSON.

## Sync Diagnosis

`POST /api/v1/diagnosis/ocr-json/sync`

Frontend-friendly alias:

`POST /api/v1/reports/from-ocr-json`

Recommended request:

```json
{
  "case_id": "demo-001",
  "top_k": 5,
  "use_multi_agent": true,
  "use_kg": true,
  "vector_sources": ["all"],
  "ocr_json": {
    "pages": [
      {
        "page_no": 1,
        "lines": [
          {"text": "ALT 85.2 U/L 参考范围 7-40"},
          {"text": "GLU 7.2 mmol/L 参考范围 3.9-6.1"},
          {"text": "LDL-C 4.1 mmol/L 参考范围 0-3.4"},
          {"text": "血压 150/95 mmHg"}
        ]
      }
    ]
  }
}
```

The backend also accepts raw OCR JSON directly, for example:

```json
{
  "pages": [
    {"lines": [{"text": "ALT 85.2 U/L"}, {"text": "GLU 7.2 mmol/L"}]}
  ]
}
```

If the OCR service already generated a pure-text JSON file, send it directly:

```json
{
  "case_id": "demo-plain-text-001",
  "plain_text": "ALT 85.2 U/L 参考范围 7-40\nGLU 7.2 mmol/L\nLDL-C 4.1 mmol/L\n血压 150/95 mmHg"
}
```

Supported OCR JSON text fields include:

- `text`
- `ocr_text`
- `full_text`
- `recognized_text`
- `line_text`
- `content`
- `pages`
- `lines`
- `blocks`
- `paragraphs`
- `results`

Response:

```json
{
  "status": "done",
  "task_id": "...",
  "report_id": "...",
  "report_path": "./data/reports/....json",
  "input_type": "ocr_json",
  "normalized_input": {
    "case_id": "demo-001",
    "source_format": "document_ocr_json",
    "line_count": 4,
    "text": "ALT 85.2 U/L 参考范围 7-40\nGLU 7.2 mmol/L ...",
    "text_preview": "ALT 85.2 U/L 参考范围 7-40..."
  },
  "report": {
    "task_id": "...",
    "case_id": "demo-001",
    "overall_risk": "...",
    "possible_diagnoses": [],
    "retrieved_cases": [],
    "kg_evidence": [],
    "agent_opinions": [],
    "critique": {},
    "summary_markdown": "...",
    "followup_questions": [],
    "entities": [],
    "raw_baseline_result": {},
    "safety_note": "This result is for course demonstration and reference only; it cannot replace a physician diagnosis."
  }
}
```

The report is persisted automatically. It can be queried again through:

- `GET /api/v1/tasks/{task_id}`
- `GET /api/v1/diagnosis/{task_id}/report`
- `GET /api/v1/history`

## Async Diagnosis

`POST /api/v1/diagnosis/ocr-json`

Uses the same request body as the sync endpoint. It returns:

```json
{
  "task_id": "...",
  "status": "pending",
  "input_type": "ocr_json",
  "normalized_input": {},
  "message": "OCR JSON diagnosis task submitted"
}
```

Use the existing task/result APIs to poll the result.

## Upload JSON File

`POST /api/v1/diagnosis/ocr-json-file/sync`

`multipart/form-data` fields:

- `file`: OCR `.json` file
- `case_id`: optional
- `top_k`: default `3`
- `use_multi_agent`: default `true`
- `use_kg`: default `true`
- `vector_sources`: optional comma-separated string, for example `all` or `ddxplus_cases,ddxplus_kg`

The response shape is the same as the sync endpoint, with an additional `file_path`.
