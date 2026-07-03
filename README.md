# MedReport AgentRAG

本项目在原 MedRAG / DDXPlus / KG-RAG / FAISS / SiliconFlow 基础上，扩展为课程设计要求的“多模态医疗报告智能解读与多 Agent 辅助诊断系统”。

当前保留原有入口：

```bat
python main.py
```

同时新增文档规定的工程化入口：

```bat
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
python frontend\app_gradio.py
```

## Anaconda 启动

如果从 Anaconda Prompt 的初始目录打开：

```bat
conda activate medrag
D:
cd D:\MedRAG-main
```

如果你的环境名是 `merge`，把第一行换成：

```bat
conda activate merge
D:
cd D:\MedRAG-main
```

## 项目结构

核心保留结构：

```text
main.py
main_MedRAG.py
KG_Retrieve.py
embedding_backend.py
engines/
baselines/
metrics/
scripts/
vector_store/
```

课程文档对齐结构：

```text
backend/
  app/
    main.py
    api/
      router_diagnosis.py
      router_upload.py
      router_task.py
      router_history.py
      router_chat.py
    schemas/
      patient_case.py
      diagnosis_result.py
      task_status.py
    services/
      pipeline.py
      task_service.py
      ocr_service.py
      report_service.py
    core/
      nlp/
      retrieval/
      kg/
      agents/
      llm/
      ocr/
frontend/
  app_gradio.py
data/
  raw/
  processed/
  df/
    train/
    test/
  kg/
  embeddings_cache/
  uploads/
  reports/
eval/
  metrics.py
```

`backend/app/core/...` 是适配层，复用现有 `engines/`、`baselines/` 里的实现，不重复写两套算法。

## 数据放置

旧路径继续支持：

```text
dataset/df/train/
dataset/df/test/
dataset/knowledge graph of DDXPlus.xlsx
```

课程文档路径也支持：

```text
data/df/train/
data/df/test/
data/kg/knowledge graph of DDXPlus.xlsx
```

如果缺少 DDXPlus JSON，请先运行：

```bat
python scripts\prepare_ddxplus_for_medrag.py
```

或者使用文档别名：

```bat
python scripts\prepare_ddxplus_for_runtime.py
```

## 后端接口

启动后端：

```bat
conda activate medrag
D:
cd D:\MedRAG-main
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

打开：

```text
http://127.0.0.1:8000/docs
```

健康检查：

```bat
curl http://127.0.0.1:8000/health
```

同步诊断：

```bat
curl -X POST http://127.0.0.1:8000/api/v1/diagnosis/text/sync ^
-H "Content-Type: application/json" ^
-d "{\"text\":\"患者男，56岁，血压150/95mmHg，LDL-C 4.2 mmol/L，GLU 7.1 mmol/L，ALT 68 U/L，自述近期乏力。\",\"top_k\":3,\"use_multi_agent\":true,\"vector_sources\":[\"all\"]}"
```

返回格式：

```json
{
  "status": "done",
  "report": {
    "task_id": "...",
    "overall_risk": "...",
    "possible_diagnoses": [],
    "retrieved_cases": [],
    "kg_evidence": [],
    "agent_opinions": [],
    "critique": {},
    "summary_markdown": "...",
    "followup_questions": [],
    "entities": [],
    "safety_note": "..."
  }
}
```

异步诊断：

```bat
curl -X POST http://127.0.0.1:8000/api/v1/diagnosis/text ^
-H "Content-Type: application/json" ^
-d "{\"text\":\"患者女，49岁，胸口烧灼感，饭后反酸，平躺加重。\",\"top_k\":3,\"use_multi_agent\":true}"
```

查询任务：

```bat
curl http://127.0.0.1:8000/api/v1/tasks/你的task_id
```

查询报告：

```bat
curl http://127.0.0.1:8000/api/v1/diagnosis/你的task_id/report
```

上传报告：

```bat
curl -X POST http://127.0.0.1:8000/api/v1/reports/upload ^
-F "file=@D:\MedRAG-main\data\raw\health_reports\sample.txt"
```

历史记录：

```bat
curl http://127.0.0.1:8000/api/v1/history
```

## 前端

新开一个 Anaconda Prompt：

```bat
conda activate medrag
D:
cd D:\MedRAG-main
python frontend\app_gradio.py
```

打开：

```text
http://127.0.0.1:7860
```

前端包含：

```text
Text Diagnosis  文本诊断
Upload          文件上传 + OCR/PDF 文本提取 + 异步分析
Report          按 Task ID 加载报告
Chat            基于已生成报告追问
History         历史报告列表
```

## Docker / Redis

Docker 一键启动后端、前端和 Redis：

```bat
docker compose up --build
```

打开：

```text
后端 API: http://127.0.0.1:8000/docs
前端界面: http://127.0.0.1:7860
```

如果不使用 Docker，也可以只在本地启 Redis，然后设置：

```bat
set REDIS_URL=redis://127.0.0.1:6379/0
```

不设置 `REDIS_URL` 时，系统自动使用内存任务缓存。

## Baseline

B0 纯 LLM/mock：

```bat
python baselines\run_b0_direct.py --text "ALT 85.2 U/L GLU 7.2 mmol/L LDL-C 4.1 mmol/L 血压 150/95 mmHg" --mock
```

B1 RAG：

```bat
python baselines\run_b1_rag.py --text "ALT 85.2 U/L GLU 7.2 mmol/L LDL-C 4.1 mmol/L 血压 150/95 mmHg" --mock --top-k 3 --output-dir storage\results
```

B1 多向量库：

```bat
python baselines\run_b1_rag.py --text "18-year-old male with fever cough sore throat and night sweats" --mock --top-k 10 --top-k-per-source 2 --vector-sources all --output-dir storage\results
```

B1 全量批处理：

```bat
python baselines\run_b1_rag.py --limit all --top-k 5 --mock --resume --output storage\results\b1_rag_results.jsonl
```

B2 KG-RAG：

```bat
python baselines\run_b2_kg_rag.py --limit all --top-k 5 --kg-top-k 10 --mock --resume --output storage\results\b2_kg_rag_results.jsonl
```

非 LLM 检索评估：

```bat
python baselines\run_retrieval_eval.py --limit all --top-k 5 --output storage\results\retrieval_eval_full.json --details-output storage\results\retrieval_eval_full_details.jsonl
```

Metrics：

```bat
python metrics\metrics_DDXPlus.py --inputs storage\results\b1_rag_results.jsonl storage\results\b2_kg_rag_results.jsonl storage\results\retrieval_eval_full_details.jsonl --output storage\metrics\metrics_summary.csv
```

NER 指标抽取评估：

```bat
python scripts\evaluate_ner.py --gold tests\fixtures\ner_eval_samples.jsonl --output storage\metrics\ner_eval_summary.json --details-csv storage\metrics\ner_eval_details.csv
```

NER/RAG 对接字段说明见：

```text
docs\rag_ner_contract.md
```

文档别名：

```bat
python eval\metrics.py --help
```

## 向量库

构建 DDXPlus 病例 FAISS：

```bat
python scripts\build_case_embeddings.py
```

构建多数据源向量库：

```bat
python scripts\build_vector_stores.py --sources ddxplus_cases ddxplus_kg pmc_patients medcase_reasoning open_patients --max-per-source 5000 --batch-size 32 --force --local
```

跨库检索：

```bat
python scripts\retrieve_multi_vector.py --query "18-year-old male with fever cough sore throat and night sweats" --sources all --top-k 10 --top-k-per-source 2 --local
```

FAISS / 多向量库 top-k 调优表：

```bat
python scripts\tune_faiss_retrieval.py --sources all --top-k-values 3 5 10 --top-k-per-source-values 2 3 5 --local --output storage\metrics\faiss_tuning_results.csv
```

## 验证

静态编译：

```bat
python -m py_compile main.py main_MedRAG.py KG_Retrieve.py embedding_backend.py
python -m py_compile backend\app\main.py backend\app\services\pipeline.py backend\app\api\router_diagnosis.py
python -m py_compile frontend\app_gradio.py scripts\build_case_embeddings.py scripts\run_api_test.py
```

Smoke tests：

```bat
python tests\smoke_test_ner.py
python tests\smoke_test_ner_noise.py
python tests\smoke_test_faiss.py
python tests\smoke_test_baseline.py
python tests\smoke_test_multi_source.py
python tests\smoke_test_kg_hybrid.py
python tests\smoke_test_chinese_kg.py
python tests\smoke_test_ocr_service.py
python tests\smoke_test_frontend_import.py
python tests\smoke_test_upload_contract.py
```

API 测试需要先启动后端，然后运行：

```bat
python scripts\run_api_test.py
```

## Git 安全

不要提交：

```text
authentication.py
.env
dataset/
data/df/
data/kg/
data/uploads/
data/reports/
external_datasets/
vector_db/
storage/results/
storage/indexes/
storage/embeddings/
models/
```

这些都是本地数据、缓存、运行结果或密钥。仓库只保留 `.gitkeep` 占位目录。

## 当前状态

已完成：

- B0/B1/B2 baseline 骨架和全量 DDXPlus 批处理。
- DDXPlus FAISS 病例检索。
- DDXPlus KG 证据检索和可选 KG 向量检索。
- 中文体检领域 KG：心血管、肝脏、内分泌/代谢、血常规、炎症风险。
- 多向量库检索：DDXPlus cases、DDXPlus KG、PMC Patients、MedCase Reasoning、Open Patients。
- NER 指标覆盖扩展、脏文本容错和 Precision/Recall/F1 评估。
- OCR/PDF 服务封装：文本 PDF 解析、图片 OCR 可选、扫描 PDF 可选渲染 OCR。
- 多 Agent 专科分析、Critique 和报告汇总规则增强。
- `backend.app.main:app` FastAPI 入口。
- `/api/v1/diagnosis/text/sync`、`/api/v1/diagnosis/text`、`/api/v1/tasks/{task_id}`、`/api/v1/reports/upload`、`/api/v1/history`、`/api/v1/chat`。
- Redis 可选任务缓存，未配置时自动回退内存缓存。
- `frontend/app_gradio.py` 多标签页前端入口。
- Dockerfile 和 docker-compose 一键启动配置。

下一阶段：

- 准备真实脱敏体检报告样例，扩大 OCR 和 NER 评估集。
- 如果演示必须识别扫描 PDF/图片，安装并测试 PaddleOCR。
- 根据老师反馈继续扩充中文 KG 节点和专科规则。
- 补充答辩 PPT、消融实验表和演示脚本。

## License

本项目继承原 MedRAG 仓库许可和引用要求。原项目代码基于 CC BY-NC 4.0，数据集和外部模型请遵守各自许可。
