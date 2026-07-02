# MedRAG — DDXPlus 差分诊断辅助系统

基于检索增强生成（RAG）与知识图谱（KG）的 DDXPlus 医疗差分诊断决策支持系统。

## 项目功能

- **病例检索（RAG）**：使用 FAISS 对训练集病例做语义检索，找到与当前患者最相似的病例，辅助 LLM 推理。
- **知识图谱增强（KG）**：利用 DDXPlus 疾病知识图谱，从症状出发定位候选疾病和相关医学关系。
- **LLM 诊断推理**：结合检索到的相似病例和知识图谱信息，由 LLM 生成结构化的诊断报告。
- **自动回退机制**：当远程 Embedding API 不可用时，自动切换到本地 `sentence-transformers` 模型，保证流程不中断。

## 环境安装

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd MedRAG-main

# 2. 创建虚拟环境（推荐）
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

**主要依赖**：`openai`、`faiss-cpu`、`numpy`、`pandas`、`sentence-transformers`、`transformers`、`torch`、`networkx`、`scikit-learn`、`openpyxl`

## 配置 API Key

本项目使用**硅基流动（SiliconFlow）**作为 LLM 和 Embedding 的 API 后端。

1. 将 `authentication.example.py` 复制为 `authentication.py`：
   ```bash
   # Windows
   copy authentication.example.py authentication.py
   # Linux / macOS
   cp authentication.example.py authentication.py
   ```

2. 编辑 `authentication.py`，将 `api_key` 替换为你自己的硅基流动 API Key：
   ```python
   api_key = "sk-your-actual-api-key"
   ```

3. （可选）如需使用 Hugging Face 模型，可填入 `hf_token`。当前版本不使用 Hugging Face 推理 API，可留空。

> ⚠️ **安全警告**：`authentication.py` 已被 `.gitignore` 忽略，**切勿**将包含真实 API Key 的 `authentication.py` 上传到公开仓库。

## 数据准备

数据集文件**不随仓库上传**，你需要自行准备 DDXPlus 原始数据。

### 1. 下载 DDXPlus 数据集

将以下文件放入 `dataset/ddxplus_raw/` 目录：

```
dataset/ddxplus_raw/
├── release_train_patients.zip      # 训练集
├── release_validate_patients.zip   # 验证集
├── release_test_patients.zip       # 测试集
├── release_evidences.json          # 证据定义
├── release_conditions.json         # 疾病条件映射
└── ...
```

### 2. 准备知识图谱文件

将 DDXPlus 知识图谱 Excel 文件放入：
```
dataset/knowledge graph of DDXPlus.xlsx
```

### 3. 运行数据转换脚本

```bash
# 少量测试（默认 1000 条训练 / 300 条测试）
python scripts/prepare_ddxplus_for_medrag.py

# 全量生成
python scripts/prepare_ddxplus_for_medrag.py --all

# 自定义数量
python scripts/prepare_ddxplus_for_medrag.py --max-train 200 --max-test 100
```

转换完成后会生成：
```
dataset/
├── AI Data Set with Categories.csv    # 总标注 CSV
├── df/
│   ├── train/
│   │   ├── participant_1.json         # 训练病例
│   │   └── ...
│   └── test/
│       ├── participant_1.json         # 测试病例
│       └── ...
```

## 运行

```bash
python main.py
```

首次运行时会：
1. 调用 SiliconFlow Embedding API 为所有训练病例生成向量（约需几分钟）。
2. 缓存向量到 `dataset/document_embeddings_*.npy`，下次运行直接加载。
3. 对前 5 个测试病例逐一执行：检索 → KG 增强 → LLM 诊断。
4. 结果保存为 `test_results_medrag_topk3_topn1_matchn5_cases5.csv`。

### 可调参数

在 `main.py` 的 `main` 部分修改：

```python
run_medrag(
    top_k=3,        # FAISS 检索的相似病例数
    top_n=1,        # KG 候选类别数
    match_n=5,      # KG 症状匹配数
    max_cases=5,    # 运行的测试病例数（调大可以跑更多）
)
```

## Embedding 回退机制

当 SiliconFlow Embedding API 调用失败时，系统会**自动回退**到本地 `sentence-transformers` 模型：

| 优先级 | 模型 | 维度 | 说明 |
|--------|------|------|------|
| 1 | `BAAI/bge-small-en-v1.5` | 384 | 约 100MB，CPU 友好，默认首选 |
| 2 | `sentence-transformers/all-MiniLM-L6-v2` | 384 | 更小兜底模型 |
| 3 | `BAAI/bge-m3` | 1024 | 与远程模型同名，需要较多内存（~5GB） |

回退逻辑：
- 远程 API 报错 → 自动下载本地模型到 `./models/embedding/`
- 自动重建 document embeddings 和 KG embeddings
- 可通过 `authentication.py` 配置本地模型参数：

```python
local_embedding_model = "BAAI/bge-small-en-v1.5"   # 首选本地模型
local_embedding_device = "cpu"                      # 默认 CPU，可改为 "cuda"
local_embedding_batch_size = 16                     # 批处理大小
local_embedding_max_memory_gb = 16.0                # 内存上限
```

## 项目结构

```
MedRAG-main/
├── main.py                     # 入口：数据检查 + 运行诊断流程
├── main_MedRAG.py              # 核心：RAG 检索 + KG 增强 + LLM 诊断
├── KG_Retrieve.py              # 知识图谱症状检索与类别推断
├── embedding_backend.py        # Embedding 后端（远程 API / 本地回退）
├── authentication.example.py   # 配置文件模板（需复制为 authentication.py）
├── requirements.txt            # Python 依赖
├── .gitignore                  # Git 忽略规则
├── LICENCE                     # 许可证
├── vector_store/               # 多向量库模块
│   ├── adapters.py             #   数据源适配器（7 种格式 → 统一 record）
│   ├── builder.py              #   FAISS 向量库构建
│   ├── retriever.py            #   单库检索
│   ├── registry.py             #   多库联合检索
│   └── utils.py                #   Embedding 函数工厂
├── scripts/
│   ├── prepare_ddxplus_for_medrag.py  # DDXPlus 原始数据 → MedRAG 格式
│   ├── build_vector_stores.py         # 构建向量库
│   ├── retrieve_multi_vector.py       # 多库检索
│   └── inspect_vector_store.py        # 检查向量库
├── dataset/                    # 数据目录（不上传 GitHub）
│   ├── df/train/               # 训练病例 JSON
│   ├── df/test/                # 测试病例 JSON
│   └── ...
├── external_datasets/          # 外部数据集（不上传 GitHub）
├── vector_db/                  # 向量库构建产物（不上传 GitHub）
├── images/                     # 论文插图
├── appendix/                   # 论文附录
└── metrics/                    # 评估指标脚本
```

## 多向量库设计 (Multi-Vector Store Design)

### 为什么不把所有数据混成一个库？

如果将所有临床文本（DDXPlus、PubMed 病例、诊断推理、KG 三元组）混入一个 FAISS 索引：

- **语义混淆**：KG 三元组（如"Bronchitis has_symptomatology cough"）和临床叙事属于完全不同的语义空间
- **维度不一致**：不同后端/模型产生的向量维度可能不同（远程 BAAI/bge-m3 1024 维 vs 本地 bge-small 384 维）
- **无法按需选择**：某些诊断场景只需搜 DDXPlus 历史病例，不需要 PubMed 文章干扰
- **增量更新困难**：加一个新数据源需要重建整个索引，而独立库只需重建新库

### 支持的向量库

| 库名 | 数据来源 | 用途 | 含诊断标签 |
|------|----------|------|:---:|
| `ddxplus_cases` | `dataset/df/train/*.json` | DDXPlus 症状-诊断相似病例检索 | ✅ |
| `ddxplus_kg` | `dataset/knowledge graph of DDXPlus.xlsx` | 知识图谱三元组检索 | ❌ |
| `pmc_patients` | `external_datasets/pmc_patients/` | PMC 临床病例摘要检索（167k+） | ❌ |
| `medcase_reasoning` | `external_datasets/medcase_reasoning/` | 诊断推理案例检索（14k+） | ✅ |
| `open_patients` | `external_datasets/open_patients/` | 多源患者描述检索（180k+） | ❌ |
| `multicare_cases` | Zenodo (DOI: 10.5281/zenodo.10079369) | 多模态病例文本检索 | ❌ |
| `synthea_records` | `external_datasets/synthea/output/` | 模拟患者记录检索 | ❌ |

> **注意**：MultiCaRe 临床文本托管在 Zenodo，Synthea 需要 JDK 运行生成器。这两种数据需要额外步骤才能用于构建向量库。

### 下载外部数据集

外部数据集**不随仓库上传**，需要自行下载并放入 `external_datasets/`：

```bash
external_datasets/
├── pmc_patients/
│   └── PMC-Patients.csv          # https://github.com/zhao-zw/PMC-Patients
├── medcase_reasoning/
│   └── medcasereasoning_core.csv # https://huggingface.co/datasets/MedCase/MedCase-Reasoning
├── open_patients/
│   └── Open-Patients.jsonl       # https://github.com/zhao-zw/Open-Patients
├── multicare_repo/               # git clone + Zenodo 下载
└── synthea/                      # git clone + ./run_synthea 生成
```

### 构建向量库

```bash
# 只构建 ddxplus_cases，100 条，强制覆盖，纯本地 embedding
python scripts/build_vector_stores.py --sources ddxplus_cases --max-per-source 100 --force --local

# 构建所有可用源（每个源最多 5000 条）
python scripts/build_vector_stores.py --sources all --max-per-source 5000

# 构建指定几个源
python scripts/build_vector_stores.py --sources ddxplus_cases ddxplus_kg pmc_patients --max-per-source 1000 --batch-size 8
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sources` | 要构建的库名（可多个），`all` 表示全部 | 必填 |
| `--max-per-source` | 每个源最大记录数 | 5000 |
| `--batch-size` | 每批 embedding 文本数 | 8 |
| `--force` | 强制覆盖已有索引 | 关闭 |
| `--local` | 仅使用本地 sentence-transformers | 关闭 |

构建产物：
```
vector_db/<source_name>/
├── index.faiss     # FAISS 向量索引
├── meta.jsonl      # 每行一条标准化 record JSON
└── config.json     # source, dim, num_records, embedding_backend, model
```

### 检索

```bash
# 查询所有可用库
python scripts/retrieve_multi_vector.py --query "70-year-old with cough, night sweats and chest pain" --top-k 10 --local

# 只在指定库中检索
python scripts/retrieve_multi_vector.py --query "fever and rash" --sources ddxplus_cases pmc_patients --top-k 10
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query` / `-q` | 查询文本 | 必填 |
| `--sources` | 检索的库名（可多个） | 全部 |
| `--top-k` | 最终返回结果数 | 10 |
| `--top-k-per-source` | 每个库返回数 | 5 |
| `--local` | 仅使用本地 embedding | 关闭 |
| `--output` | 保存结果到 JSON 文件 | 无 |
| `--verbose` | 打印完整字段 | 关闭 |

### 检查向量库

```bash
python scripts/inspect_vector_store.py --source ddxplus_cases
python scripts/inspect_vector_store.py --source pmc_patients --samples 10
```

### 多库检索 Python API

```python
from vector_store.registry import MultiVectorRetriever
from vector_store.utils import create_embedding_fn, create_query_embedding_fn

# 创建 embedding 函数
embed_fn = create_embedding_fn(force_local=True)
query_fn = create_query_embedding_fn(embed_fn)

# 初始化检索器
retriever = MultiVectorRetriever(base_dir="./vector_db")
print("Available sources:", retriever.sources)

# 联合检索
results = retriever.search(
    query="fever, cough, night sweats",
    embedding_fn=query_fn,
    sources=["ddxplus_cases", "pmc_patients"],
    top_k_per_source=5,
    final_top_k=10,
)

for r in results:
    print(f"[{r['score']:.4f}] [{r['source']}] {r['title']}: {r['text'][:100]}...")
```

### 数据流

```
external_datasets/     adapters.py       builder.py        vector_db/
┌───────────────┐    ┌──────────┐      ┌──────────┐      ┌──────────────┐
│ PMC-Patients  │ -> │ adapter  │ ->   │  build   │ ->   │ index.faiss  │
│ MedCase       │    │ per src  │      │  FAISS   │      │ meta.jsonl   │
│ Open-Patients │    └──────────┘      │  store   │      │ config.json  │
│ DDXPlus       │                      └──────────┘      └──────────────┘
└───────────────┘                           │
                                        embedding_fn
                                            │
                                  embedding_backend.py
                                  (SiliconFlow / local)
```

### Embedding 一致性

每个向量库的 `config.json` 记录了构建时使用的 embedding 后端和模型名。`embedding_backend.py` 的全局状态确保同一进程中 query embedding 和 document embedding 使用相同的模型和维度，避免 FAISS 维度不匹配错误。如果切换了 embedding 后端（如远程 API 失败切到本地），已缓存的向量库需要用 `--force` 重建。

## 许可证

本项目代码基于 [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/) 许可。

Copyright (c) 2024 Xuejiao Zhao. Contact: xuejiaozhao_snow@foxmail.com

## 致谢

- [DDXPlus](https://github.com/mila-iqia/ddxplus) — 合成医疗诊断数据集
- [SNOWTEAM2023/MedRAG](https://github.com/SNOWTEAM2023/MedRAG) — 原始 RAG 框架
- [SiliconFlow](https://siliconflow.cn) — LLM & Embedding API 服务
- [sentence-transformers](https://www.sbert.net/) — 本地 Embedding 模型

## 课程阶段验收说明：全量 DDXPlus RAG / KG-RAG

本仓库在保留原有 `python main.py` 运行方式的基础上，新增了面向课程设计的模块化骨架与全量 DDXPlus 批量评估能力。当前重点完成的是：

- `dataset/df/train` 全量构建 FAISS 病例检索库。
- `dataset/knowledge graph of DDXPlus.xlsx` 全量读取并用于 KG evidence 检索。
- `dataset/df/test` 全量批量运行 B1 / B2。
- B1 RAG baseline 支持 `--limit all`、`--resume`、失败不中断、逐条追加 JSONL。
- B2 KG-RAG baseline 复用 B1 FAISS 检索，并增加 KG evidence 与 Agent 骨架输出。
- 新增非 LLM 检索评估脚本，支持快速计算 Recall@1 / Recall@3 / Recall@5。
- `metrics/metrics_DDXPlus.py` 支持读取 JSON、JSONL、CSV 并输出汇总 CSV。

### 环境切换

如果从 Anaconda Prompt 的初始目录打开，请先切换环境和项目目录：

```bat
conda activate medrag
cd /d D:\MedRAG-main
```

如果你的环境名称是 `merge`，则把第一行改成：

```bat
conda activate merge
cd /d D:\MedRAG-main
```

### 数据准备

如果缺少 `dataset/df/train` 或 `dataset/df/test`，请先运行：

```bat
python scripts\prepare_ddxplus_for_medrag.py
```

KG 文件应放在：

```text
dataset\knowledge graph of DDXPlus.xlsx
```

### 代码正确性验证

静态编译检查：

```bat
python -m py_compile main.py main_MedRAG.py KG_Retrieve.py embedding_backend.py engines\retrieval\embedding_engine.py engines\retrieval\faiss_retriever.py baselines\run_b1_rag.py baselines\run_b2_kg_rag.py baselines\run_retrieval_eval.py metrics\metrics_DDXPlus.py
```

Smoke tests：

```bat
python tests\smoke_test_ner.py
python tests\smoke_test_faiss.py
python tests\smoke_test_baseline.py
```

### B1 全量 RAG baseline

Mock 模式不调用真实 LLM，适合课程验收和流程验证：

```bat
python baselines\run_b1_rag.py --limit all --top-k 5 --mock --resume --output storage\results\b1_rag_results.jsonl
```

输出：

```text
storage\results\b1_rag_results.jsonl
```

### B2 全量 KG-RAG baseline

```bat
python baselines\run_b2_kg_rag.py --limit all --top-k 5 --kg-top-k 10 --mock --resume --output storage\results\b2_kg_rag_results.jsonl
```

输出：

```text
storage\results\b2_kg_rag_results.jsonl
```

### 非 LLM 检索评估

该脚本不调用 LLM，只评估 ground truth 是否命中 top-k retrieved cases：

```bat
python baselines\run_retrieval_eval.py --limit all --top-k 5 --output storage\results\retrieval_eval_full.json --details-output storage\results\retrieval_eval_full_details.jsonl
```

已验证的全量 test 结果：

```text
total_cases = 300
Recall@1 = 0.9567
Recall@3 = 0.9833
Recall@5 = 0.99
```

### Metrics 汇总

```bat
python metrics\metrics_DDXPlus.py --inputs storage\results\b1_rag_results.jsonl storage\results\b2_kg_rag_results.jsonl storage\results\retrieval_eval_full_details.jsonl --output storage\metrics\metrics_summary.csv
```

输出：

```text
storage\metrics\metrics_summary.csv
```

### 缓存与 Git 提交注意事项

运行时会自动生成以下缓存和结果文件：

```text
storage\embeddings\ddxplus_cases.npy
storage\embeddings\ddxplus_cases_metadata.jsonl
storage\indexes\ddxplus_cases.faiss
storage\indexes\ddxplus_cases_metadata.jsonl
storage\results\*.jsonl
storage\metrics\*.csv
```

这些运行产物已被 `.gitignore` 忽略，不建议提交到 GitHub。仓库只保留 `.gitkeep` 用于占位目录。

不要提交真实 API Key。`authentication.py` 必须保持被 `.gitignore` 忽略。
