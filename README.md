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
├── scripts/
│   └── prepare_ddxplus_for_medrag.py  # DDXPlus 原始数据 → MedRAG 格式
├── dataset/                    # 数据目录（不上传 GitHub）
│   ├── df/train/               # 训练病例 JSON
│   ├── df/test/                # 测试病例 JSON
│   └── ...
├── images/                     # 论文插图
├── appendix/                   # 论文附录
└── metrics/                    # 评估指标脚本
```

## 许可证

本项目代码基于 [CC BY-NC 4.0](http://creativecommons.org/licenses/by-nc/4.0/) 许可。

Copyright (c) 2024 Xuejiao Zhao. Contact: xuejiaozhao_snow@foxmail.com

## 致谢

- [DDXPlus](https://github.com/mila-iqia/ddxplus) — 合成医疗诊断数据集
- [SNOWTEAM2023/MedRAG](https://github.com/SNOWTEAM2023/MedRAG) — 原始 RAG 框架
- [SiliconFlow](https://siliconflow.cn) — LLM & Embedding API 服务
- [sentence-transformers](https://www.sbert.net/) — 本地 Embedding 模型
