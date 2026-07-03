# RAG 与 NER 模块对接契约

本文档用于周子烨负责的 NER + FAISS/RAG 模块与 KG、多 Agent、后端、Baseline 评估模块对接。

## 1. NER 输入输出

入口函数：

```python
from engines.ner.medical_ner import extract_medical_entities, entities_to_query_text
```

输入：

```python
text: str
```

输出：

```json
[
  {
    "name": "ALT",
    "value": 85.2,
    "unit": "U/L",
    "ref_low": 7.0,
    "ref_high": 40.0,
    "is_abnormal": true,
    "original_text": "ALT 85.2 U/L 参考范围 7-40"
  }
]
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `name` | string | 标准化指标名，用于 KG L4 特征节点、Agent prompt、RAG query |
| `value` | number | 抽取出的数值 |
| `unit` | string | 标准化单位 |
| `ref_low` | number/null | 参考范围下限 |
| `ref_high` | number/null | 参考范围上限 |
| `is_abnormal` | bool | 是否超出参考范围 |
| `original_text` | string | 触发该实体的原始片段 |

## 2. 当前标准指标名

当前规则覆盖：

```text
ALT, AST, GGT, ALP, TBIL, DBIL, IBIL, ALB,
GLU, HbA1c, TSH, FT3, FT4,
TC, TG, LDL-C, HDL-C,
Cr, BUN, 尿酸,
WBC, RBC, HGB, PLT,
收缩压, 舒张压, BMI, 心率
```

KG 模块建议直接使用这些名称作为体检指标型 L4 节点名，或维护一张别名映射表把 KG 节点映射到这些标准名。

## 3. FAISS / 多向量检索输出

入口：

```python
from engines.retrieval.faiss_retriever import FaissCaseRetriever
from engines.retrieval.multi_source_retriever import MultiSourceRetriever
```

`retrieved_cases` 标准字段：

```json
{
  "case_id": "780",
  "source": "ddxplus_cases",
  "title": "Tuberculosis case, 10M",
  "diagnosis": "Tuberculosis",
  "similarity": 0.724,
  "raw_text": "Age: 10 Sex: M Symptoms: ...",
  "metadata": {}
}
```

说明：

| 字段 | 说明 |
|---|---|
| `case_id` | 病例或向量记录 ID |
| `source` | 多向量库来源，单库 FAISS 可为空 |
| `title` | 向量记录标题，可能为空 |
| `diagnosis` | 病例诊断标签 |
| `similarity` | 归一化向量相似度 |
| `raw_text` | 检索命中的原始文本 |
| `metadata` | 来源数据的补充字段 |

## 4. KG 证据输出

后端会合并两类 KG：

```text
resources/chinese_medical_kg.json        中文体检风险 KG
data/kg/knowledge graph of DDXPlus.xlsx  DDXPlus 疾病症状 KG
```

`kg_evidence` 标准字段：

```json
{
  "head": "ALT升高",
  "relation": "supports",
  "tail": "肝细胞损伤风险",
  "source": "Chinese_Physical_Exam_KG",
  "relation_category": "test_indicator",
  "score": 10.0,
  "retrieval_method": "indicator_rule",
  "indicator": "ALT",
  "value": 85.2,
  "unit": "U/L",
  "neighbors": []
}
```

DDXPlus KG 证据也遵循 `head/relation/tail/source/score/retrieval_method/neighbors` 这组字段。

## 5. Agent 输出

`agent_opinions` 标准字段：

```json
{
  "specialty": "cardiovascular",
  "agent_name": "cardiovascular",
  "risk_level": "medium",
  "confidence": 0.78,
  "diagnosis": ["高血压风险", "血脂异常风险"],
  "evidence": ["收缩压=150 mmHg ..."],
  "matched_indicators": [],
  "kg_evidence_count": 2,
  "retrieved_case_count": 5,
  "suggestion": "建议复测血压和血脂..."
}
```

当前专科 Agent：

```text
cardiovascular  心血管
liver           肝脏
endocrine       内分泌/代谢
hematology      血常规/炎症
```

## 6. 后端调用边界

后端实时流程通过：

```python
backend.app.core.nlp.entity_extractor.EntityExtractor
backend.app.core.retrieval.case_repository.CaseRepository
backend.app.services.pipeline.DiagnosisPipeline
```

调用 NER 和检索。模块不依赖 PaddleOCR、Redis、Gradio，后端可以安全 import。

## 7. Baseline / 消融实验边界

B1 和“A2 去 KG，仅向量 RAG”应直接调用：

```python
baselines/run_b1_rag.py
engines/retrieval/faiss_retriever.py
engines/retrieval/multi_source_retriever.py
```

FAISS 检索可以单独运行，不要求 KG 先完成。

推荐命令：

```bat
python baselines\run_b1_rag.py --text "ALT 85.2 U/L GLU 7.2 mmol/L LDL-C 4.1 mmol/L 血压 150/95 mmHg" --mock --top-k 3
python scripts\evaluate_ner.py
python scripts\tune_faiss_retrieval.py --sources all --top-k-values 3 5 10 --top-k-per-source-values 2 3 5 --local
```
