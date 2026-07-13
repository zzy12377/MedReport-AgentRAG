# MedRAG-basic 说明文档

## 1. 项目简介

`MedRAG-basic` 是从开发版 `MedRAG-main` 中整理出来的答辩演示版。

它保留了后端运行、中文数据、中文知识图谱、病例向量库、Embedding 模型缓存和启动脚本，删除了翻译脚本、数据构建脚本、备份文件、测试文件等非运行必需内容。

一句话概括：

```text
输入体检/OCR JSON → 抽取医学指标 → 检索相似病例 → 查询知识图谱 → 多 Agent 综合分析 → 输出 Markdown 医疗报告
```

本项目仅用于课程演示和辅助参考，不能替代医生诊断。

## 2. 当前目录结构

```text
D:\MedRAG-basic
├── backend
├── data
├── data_zh
├── engines
├── models
├── storage
├── vector_db_zh
├── vector_store
├── .env
├── embedding_backend.py
├── requirements.txt
├── start_backend_zh.bat
└── start_backend_zh_apifox.bat
```

目录说明：

| 路径 | 作用 |
|---|---|
| `backend` | FastAPI 后端接口和业务流程 |
| `backend/app/main.py` | 后端应用入口 |
| `backend/app/api/router_diagnosis.py` | 诊断和报告生成接口 |
| `backend/app/services/pipeline.py` | 核心诊断流程调度 |
| `backend/app/services/ocr_json_service.py` | OCR JSON / 结构化 JSON 输入标准化 |
| `backend/app/services/report_service.py` | 报告保存 |
| `engines` | 检索、知识图谱、NER、多 Agent、LLM 等算法模块 |
| `engines/agents` | 多 Agent 分析模块 |
| `engines/retrieval` | FAISS 相似病例检索模块 |
| `engines/kg` | 知识图谱检索模块 |
| `engines/ner` | 医学实体和指标抽取模块 |
| `vector_store` | 向量库构建、加载和检索工具 |
| `data_zh` | 中文病例和中文知识图谱数据 |
| `vector_db_zh` | 外部中文病例向量库 |
| `storage/indexes` | 内部病例 FAISS 向量索引 |
| `models/embedding` | 本地 Embedding 模型缓存 |
| `data/reports` | 接口运行后保存的报告 JSON |
| `.env` | 本地运行配置，不建议公开展示 |
| `requirements.txt` | Python 依赖 |
| `start_backend_zh.bat` | 中文后端启动脚本 |
| `start_backend_zh_apifox.bat` | Apifox 本地测试启动脚本 |

## 3. 当前数据规模

答辩演示版当前主要数据如下：

| 数据类型 | 路径 | 数量 |
|---|---|---:|
| 中文训练病例 | `data_zh/df/train` | 1000 条 |
| 中文测试病例 | `data_zh/df/test` | 300 条 |
| 中文知识图谱三元组 | `data_zh/kg/combined_kg_zh.jsonl` | 17732 条 |
| 内部病例向量索引元数据 | `storage/indexes/ddxplus_cases_zh_metadata.jsonl` | 1000 条 |
| 外部病例库 `medcase_reasoning` | `vector_db_zh/medcase_reasoning/meta.jsonl` | 2300 条 |
| 外部病例库 `open_patients` | `vector_db_zh/open_patients/meta.jsonl` | 800 条 |

可检索病例证据总数：

```text
内部病例 1000 条 + 外部病例 2300 条 + 外部病例 800 条 = 4100 条
```

另外还有：

```text
中文知识图谱：17732 条三元组
中文测试病例：300 条
```

## 4. 运行环境

推荐使用 Anaconda 环境：

```bat
conda activate medrag
```

如果还没有安装依赖，可以在项目目录下执行：

```bat
cd /d D:\MedRAG-basic
pip install -r requirements.txt
```

注意：

- 推荐使用 Windows 的 Anaconda Prompt 或 CMD 运行。
- `.env` 中可能包含模型 API 配置，不要在答辩 PPT 或公开仓库中展示。
- 如果没有配置可用的大模型 API，系统会使用 fallback / mock 输出，核心检索流程仍可运行。

## 5. 启动后端

进入项目目录：

```bat
cd /d D:\MedRAG-basic
```

启动 Apifox 测试版后端：

```bat
start_backend_zh.bat
```

启动后默认地址：

```text
http://0.0.0.0:8000
```

OpenAPI 文档地址：

```text
http://0.0.0.0:8000/docs
```

健康检查接口：

```text
GET http://0.0.0.0:8000/health
```

## 6. 推荐测试接口

推荐使用 Markdown 报告接口：

```text
POST http://0.0.0.0:8000/api/v1/reports/from-ocr-json/markdown
```

请求体示例：

```json
{
  "case_id": "flow-test-001",
  "name": "张三",
  "gender": "男",
  "age": 28,
  "height": {
    "value": 175.0,
    "unit": "cm"
  },
  "weight": {
    "value": 68.0,
    "unit": "kg"
  },
  "blood_pressure": {
    "systolic": 150,
    "diastolic": 95,
    "unit": "mmHg"
  },
  "heart_rate": {
    "value": 72,
    "unit": "bpm"
  },
  "conclusion": "各项指标基本正常，建议保持规律作息、均衡饮食、适量运动，每年定期体检。"
}
```

主要返回字段：

| 字段 | 说明 |
|---|---|
| `status` | 任务状态 |
| `report_id` | 报告 ID |
| `report_path` | 报告保存路径 |
| `format` | 输出格式 |
| `report_text` | Markdown 医疗报告正文 |

## 7. 核心流程

系统核心流程如下：

```text
用户输入 JSON
    ↓
FastAPI 接口接收
    ↓
OCR JSON / 结构化 JSON 标准化
    ↓
医学实体和指标抽取
    ↓
B0：规则直接判断
    ↓
B1：相似病例 RAG 检索
    ↓
B2：知识图谱 + 多 Agent 综合分析
    ↓
LLMGateway / fallback 生成摘要
    ↓
Markdown 报告输出
    ↓
保存 JSON 报告
```

## 8. B0 / B1 / B2 说明

### B0：直接指标判断

B0 只使用输入中的结构化指标，不检索病例，也不查询知识图谱。

例如：

```text
收缩压 ≥ 140 或舒张压 ≥ 90
→ 判断为血压升高 / 高血压风险
```

特点：

- 稳定
- 便宜
- 可解释
- 可作为 fallback

### B1：相似病例 RAG

B1 使用 Embedding + FAISS 检索相似病例。

流程：

```text
输入病例文本
→ Embedding 模型转向量
→ FAISS 检索相似病例
→ 返回 Top-K 病例证据
```

使用的数据：

```text
storage/indexes/ddxplus_cases_zh.faiss
vector_db_zh/medcase_reasoning/index.faiss
vector_db_zh/open_patients/index.faiss
```

作用：

- 给诊断提供历史病例参考
- 减少模型凭空生成
- 增强报告可解释性

### B2：知识图谱 + 多 Agent 综合

B2 综合以下信息：

```text
输入指标
相似病例
知识图谱证据
专科 Agent 意见
```

知识图谱三元组示例：

```text
疾病 --has_symptom--> 症状
疾病 --has_exposure--> 暴露因素
疾病 --is_a--> 疾病类别
```

作用：

- 补充医学关系证据
- 提供疾病和症状之间的结构化解释
- 支持多 Agent 综合判断

## 9. 多 Agent 说明

多 Agent 模块位于：

```text
engines/agents
```

主要 Agent：

| Agent | 作用 |
|---|---|
| 心血管 Agent | 关注血压、心率、胸痛、心血管风险 |
| 肝脏 Agent | 关注肝功能和相关风险 |
| 内分泌 Agent | 关注代谢、血糖、肥胖等风险 |
| Summary Agent | 汇总专科意见 |
| Critique Agent | 复核和风险提示 |
| LLM-MDT Agent | 可选的大模型多学科会诊分析 |

输出字段：

```text
agent_opinions
critique
mdt_report
```

## 10. LLMGateway 说明

本项目不在业务代码中写死某个大模型，而是通过 LLMGateway 统一调用。

结构：

```text
业务代码
    ↓
LLMGateway
    ↓
OpenAI-compatible API / DeepSeek / Ollama / Qwen / fallback
```

好处：

- 可以切换模型
- 可以使用本地 Qwen
- 可以使用远程 API
- 没有模型时可 fallback
- 不影响主流程代码

## 11. 输出报告结构

接口最终返回 Markdown 报告，通常包括：

```text
## 医疗检测报告

### 一、检测结论
### 二、知识图谱对应疾病症状
### 三、B0 / B1 / B2 匹配率
### 四、相似病例证据
### 五、知识图谱证据明细
### 六、专科 Agent 意见
### 七、模型摘要
### 八、安全提示
```

报告会同时保存为 JSON 文件：

```text
data/reports/{report_id}.json
```

## 12. 与原始 MedRAG 的区别

原始 MedRAG 更像论文实验脚本：

```text
读取固定测试病例
→ 检索相似病例
→ 查询 Excel 知识图谱
→ 拼 prompt
→ 调用大模型
→ 保存 CSV 实验结果
```

当前 `MedRAG-basic` 是工程化后的中文后端系统：

```text
HTTP 接口输入
→ 中文数据处理
→ B0 / B1 / B2 分层分析
→ 中文 KG 检索
→ 多 Agent 综合
→ Markdown 报告输出
```

主要新增：

- FastAPI 后端接口
- Apifox 本地测试
- 中文病例数据
- 中文知识图谱
- 外部病例向量库
- B0 / B1 / B2 分层证据
- 多 Agent 分析
- LLMGateway 模型网关
- Markdown 医疗报告

## 13. 答辩推荐表述

可以这样介绍项目：

```text
本项目基于 MedRAG 思想，将原始论文实验代码工程化为一个中文医学 RAG 后端系统。
系统首先接收体检或 OCR JSON 输入，经过标准化和医学指标抽取后，分别进行 B0 规则判断、B1 相似病例检索和 B2 知识图谱加多 Agent 综合分析。
最后通过 LLMGateway 调用本地或远程模型生成摘要，并输出 Markdown 医疗报告。
系统的核心特点是结果不是单纯由大模型生成，而是结合了输入指标、病例库、知识图谱和多 Agent 证据，因此具有更好的可解释性。
```

一句话总结：

```text
这是一个“规则 + 病例 RAG + 知识图谱 + 多 Agent + 大模型总结”的中文医疗辅助分析系统。
```

## 14. 注意事项

- 本系统仅用于课程演示和辅助参考，不能替代医生诊断。
- `.env` 可能包含 API Key，不要公开展示。
- 如果启动时报缺少依赖，先执行 `pip install -r requirements.txt`。
- 如果 LLM API 未配置，报告中的模型摘要可能使用 fallback 输出。
- 如果端口 `8000` 被占用，可以修改环境变量 `BACKEND_PORT` 后再启动。
- `data/reports` 中的报告文件是运行时生成结果，可以按需清理。

