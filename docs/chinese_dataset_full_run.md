# 中文数据镜像与全量运行说明

本文档用于把 DDXPlus 原始数据转换为中文检索镜像，并基于中文镜像全量构建病例 FAISS 库、运行 B1/B2 和 metrics。

## 1. 生成中文镜像数据

从 Anaconda 初始终端开始：

```bat
conda activate medrag
D:
cd D:\MedRAG-main
python scripts\build_ddxplus_zh_dataset.py --dataset-dir dataset --output-dir data_zh --clear
```

生成后会得到：

```text
data_zh\df\train
data_zh\df\test
data_zh\kg\ddxplus_kg_zh.jsonl
data_zh\README.md
```

说明：

- 原始 `dataset` 不会被覆盖。
- `Text_zh`、`Symptoms_zh`、`Diagnosis_zh` 用于中文检索和中文展示。
- `Diagnosis_original`、`Text_original` 等字段用于复现实验和 Recall 评估。

## 2. 构建中文 FAISS 病例检索库

```bat
conda activate medrag
D:
cd D:\MedRAG-main
python scripts\build_case_embeddings.py --train-dir data_zh\df\train --force
```

中文索引默认保存到：

```text
storage\indexes\ddxplus_cases_zh.faiss
storage\indexes\ddxplus_cases_zh_metadata.jsonl
storage\embeddings\ddxplus_cases_zh.npy
storage\embeddings\ddxplus_cases_zh_metadata.jsonl
```

## 3. 使用中文数据启动后端

```bat
conda activate medrag
D:
cd D:\MedRAG-main
set USE_ZH_DATA=true
set FORCE_MOCK_LLM=true
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

前端推荐调用：

```http
POST http://127.0.0.1:8000/api/v1/reports/from-ocr-json/simple
```

页面展示字段：

```text
report_text
```

## 4. 全量非 LLM 检索评估

```bat
conda activate medrag
D:
cd D:\MedRAG-main
python baselines\run_retrieval_eval.py --test-dir data_zh\df\test --limit all --top-k 5 --output storage\results\retrieval_eval_zh_full.json --details-output storage\results\retrieval_eval_zh_full_details.jsonl
```

## 5. 全量 B1 RAG

```bat
conda activate medrag
D:
cd D:\MedRAG-main
python baselines\run_b1_rag.py --test-dir data_zh\df\test --limit all --top-k 5 --mock --resume --output storage\results\b1_rag_zh_results.jsonl
```

## 6. 全量 B2 KG-RAG

```bat
conda activate medrag
D:
cd D:\MedRAG-main
set DATA_ZH_KG_PATH=.\data_zh\kg\ddxplus_kg_zh.jsonl
python baselines\run_b2_kg_rag.py --test-dir data_zh\df\test --limit all --top-k 5 --kg-top-k 8 --mock --resume --output storage\results\b2_kg_rag_zh_results.jsonl
```

## 7. 生成 metrics 对比表

```bat
conda activate medrag
D:
cd D:\MedRAG-main
python metrics\metrics_DDXPlus.py --inputs storage\results\b1_rag_zh_results.jsonl storage\results\b2_kg_rag_zh_results.jsonl storage\results\retrieval_eval_zh_full_details.jsonl --output storage\metrics\metrics_summary_zh.csv
```

## 注意

当前中文镜像使用离线词典和规则生成，适合课程演示、中文报告展示和中文检索实验。若要医学逐句高质量翻译，可以后续接入 Qwen、ChatGLM、Ollama 或云端 LLM，但仍建议保留 `*_original` 字段，不覆盖原始 DDXPlus。
