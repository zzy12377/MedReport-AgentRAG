# 后端 OCR JSON 生成报告接口文档

本文档用于前后端对接：FastAPI 后端接收 OCR 识别后的 JSON，提取报告文本，执行 NER、相似病例检索、KG 证据检索和报告生成，最后返回结构化诊断报告。

## 1. 服务地址

本地开发默认地址：

```text
http://127.0.0.1:8000
```

API 前缀：

```text
/api/v1
```

启动命令：

```bat
(base) C:\Users\john smith>conda activate medrag
(medrag) C:\Users\john smith>D:
(medrag) D:\>cd D:\MedRAG-main
(medrag) D:\MedRAG-main>uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

健康检查：

```http
GET /health
```

返回：

```json
{
  "status": "ok",
  "service": "backend.app.main"
}
```

## 2. 推荐主接口：OCR JSON 生成报告

```http
POST /api/v1/reports/from-ocr-json
Content-Type: application/json
```

这个接口是前端推荐使用的主接口。它会同步生成报告、保存报告文件，并返回报告内容。

### 2.1 请求体格式 A：包装后的 OCR JSON

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

### 2.2 请求体格式 B：纯文本 JSON

如果 OCR 模块已经把报告转换成纯文本 JSON，可以直接传：

```json
{
  "case_id": "demo-plain-text-001",
  "top_k": 5,
  "plain_text": "ALT 85.2 U/L 参考范围 7-40\nGLU 7.2 mmol/L\nLDL-C 4.1 mmol/L\n血压 150/95 mmHg"
}
```

也支持字段名：

```text
text, ocr_text, full_text, plain_text, raw_text, report_text, recognized_text, line_text, content
```

### 2.3 请求字段说明

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---:|---:|---:|---|
| case_id | string | 否 | null | 前端病例编号，原样写入返回报告 |
| top_k | integer | 否 | 3 | 相似病例检索数量 |
| use_multi_agent | boolean | 否 | true | 是否启用多 Agent 分析 |
| use_kg | boolean | 否 | true | 是否启用知识图谱证据 |
| vector_sources | string[] | 否 | null | 向量库来源，例如 `["all"]`、`["ddxplus_cases"]` |
| ocr_json | object | 否 | null | OCR 原始 JSON；若传包装格式则建议使用 |
| plain_text/text/ocr_text | string | 否 | null | OCR 后纯文本内容 |

### 2.4 成功返回

```json
{
  "status": "done",
  "task_id": "7de7a1b9-xxxx",
  "report_id": "7de7a1b9-xxxx",
  "report_path": "./data/reports/7de7a1b9-xxxx.json",
  "input_type": "ocr_json",
  "normalized_input": {
    "case_id": "demo-001",
    "source_format": "document_ocr_json",
    "line_count": 4,
    "text": "ALT 85.2 U/L 参考范围 7-40\nGLU 7.2 mmol/L ...",
    "text_preview": "ALT 85.2 U/L 参考范围 7-40..."
  },
  "report": {
    "task_id": "7de7a1b9-xxxx",
    "case_id": "demo-001",
    "overall_risk": "medium",
    "possible_diagnoses": [],
    "retrieved_cases": [],
    "kg_evidence": [],
    "agent_opinions": [],
    "critique": {},
    "summary_markdown": "## Diagnosis Report...",
    "followup_questions": [],
    "entities": [],
    "raw_baseline_result": {},
    "safety_note": "This result is for course demonstration and reference only; it cannot replace a physician diagnosis.",
    "input_type": "ocr_json",
    "normalized_input": {}
  }
}
```

前端主要展示字段：

| 字段 | 用途 |
|---|---|
| normalized_input.text_preview | 展示 OCR 解析出的文本预览 |
| report.entities | 展示 NER 抽取出的指标 |
| report.retrieved_cases | 展示相似病例 |
| report.kg_evidence | 展示知识图谱证据 |
| report.agent_opinions | 展示各专科 Agent 结果 |
| report.critique | 展示冲突检测和置信度校准 |
| report.summary_markdown | 展示最终报告正文 |
| report_path | 后端本地保存路径 |

### 2.5 失败返回

当 OCR JSON 中没有可提取文本时：

```json
{
  "detail": "OCR JSON 中没有可用于诊断的文本。请确认包含 text、ocr_text、plain_text、pages、lines、blocks 或 results 字段。"
}
```

HTTP 状态码：

```text
400
```

## 3. 等价同步接口

```http
POST /api/v1/diagnosis/ocr-json/sync
Content-Type: application/json
```

功能与 `/api/v1/reports/from-ocr-json` 相同，保留它是为了兼容诊断模块命名。

## 4. 异步 OCR JSON 诊断接口

```http
POST /api/v1/diagnosis/ocr-json
Content-Type: application/json
```

请求体与主接口相同。

返回：

```json
{
  "task_id": "7de7a1b9-xxxx",
  "status": "pending",
  "input_type": "ocr_json",
  "normalized_input": {
    "case_id": "demo-001",
    "source_format": "document_ocr_json",
    "line_count": 4,
    "text": "...",
    "text_preview": "..."
  },
  "message": "OCR JSON diagnosis task submitted"
}
```

前端拿到 `task_id` 后轮询：

```http
GET /api/v1/tasks/{task_id}
```

任务完成后也可以读取报告：

```http
GET /api/v1/diagnosis/{task_id}/report
```

## 5. 上传 JSON 文件生成报告

```http
POST /api/v1/diagnosis/ocr-json-file/sync
Content-Type: multipart/form-data
```

表单字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---:|---:|---|
| file | file | 是 | OCR 输出的 `.json` 文件 |
| case_id | string | 否 | 病例编号 |
| top_k | integer | 否 | 相似病例数量，默认 3 |
| use_multi_agent | boolean | 否 | 是否启用多 Agent |
| use_kg | boolean | 否 | 是否启用 KG |
| vector_sources | string | 否 | 逗号分隔，例如 `all` 或 `ddxplus_cases,ddxplus_kg` |

返回结构与主接口一致，额外包含：

```json
{
  "file_path": "./data/uploads/xxxx_ocr.json"
}
```

## 6. 文本诊断接口

同步：

```http
POST /api/v1/diagnosis/text/sync
Content-Type: application/json
```

请求：

```json
{
  "text": "ALT 85.2 U/L GLU 7.2 mmol/L LDL-C 4.1 mmol/L 血压 150/95 mmHg",
  "top_k": 5,
  "use_multi_agent": true,
  "use_kg": true,
  "vector_sources": ["all"]
}
```

异步：

```http
POST /api/v1/diagnosis/text
```

## 7. 查询报告和历史记录

查询任务状态：

```http
GET /api/v1/tasks/{task_id}
```

查询报告：

```http
GET /api/v1/diagnosis/{task_id}/report
```

查询历史报告：

```http
GET /api/v1/history
```

查询单条历史：

```http
GET /api/v1/history/{task_id}
```

报告默认保存目录：

```text
data/reports/
```

## 8. 前端 fetch 示例

```javascript
const res = await fetch("http://127.0.0.1:8000/api/v1/reports/from-ocr-json", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    case_id: "demo-001",
    top_k: 5,
    use_multi_agent: true,
    use_kg: true,
    vector_sources: ["all"],
    plain_text: "ALT 85.2 U/L\nGLU 7.2 mmol/L\nLDL-C 4.1 mmol/L\n血压 150/95 mmHg"
  })
});

const data = await res.json();
console.log(data.report.summary_markdown);
```

## 9. curl 示例

```bat
curl -X POST "http://127.0.0.1:8000/api/v1/reports/from-ocr-json" ^
  -H "Content-Type: application/json" ^
  -d "{\"case_id\":\"demo-001\",\"plain_text\":\"ALT 85.2 U/L\nGLU 7.2 mmol/L\nLDL-C 4.1 mmol/L\n血压 150/95 mmHg\",\"top_k\":5}"
```

## 10. 验证命令

```bat
(base) C:\Users\john smith>conda activate medrag
(medrag) C:\Users\john smith>D:
(medrag) D:\>cd D:\MedRAG-main
(medrag) D:\MedRAG-main>python -m py_compile backend/app/api/router_diagnosis.py backend/app/services/ocr_json_service.py backend/app/services/task_service.py
(medrag) D:\MedRAG-main>python tests\smoke_test_ocr_json_api.py
```

如果提示缺少 FastAPI：

```bat
(medrag) D:\MedRAG-main>pip install fastapi uvicorn python-multipart
```

## 11. 数据流

```text
OCR JSON / pure text JSON
-> /api/v1/reports/from-ocr-json
-> normalize_ocr_json
-> DiagnosisPipeline
-> medical NER
-> vector retrieval / FAISS
-> KG evidence
-> optional multi-agent analysis
-> summary report
-> save to data/reports/{task_id}.json
-> return JSON response
```
