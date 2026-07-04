# -*- coding: utf-8 -*-
"""Real-time diagnosis pipeline used by the FastAPI interface."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional

from backend.app.config.settings import settings
from backend.app.core.kg.kg_subgraph_retriever import KGSubgraphRetriever
from backend.app.core.llm.llm_gateway import LLMGateway
from backend.app.core.nlp.entity_extractor import EntityExtractor
from backend.app.core.retrieval.case_repository import CaseRepository
from backend.app.schemas.diagnosis_result import FinalReport
from engines.agents.agent_pipeline import run_agent_pipeline
from engines.llm.llm_gateway import extract_prediction
from engines.ner.medical_ner import entities_to_query_text


def _dump_model(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


class DiagnosisPipeline:
    def __init__(self) -> None:
        self.extractor = EntityExtractor()
        self.case_repo = CaseRepository.get_instance()
        self.kg_retriever = KGSubgraphRetriever()
        self.llm = LLMGateway.get_instance(mock=settings.force_mock_llm)

    async def run(
        self,
        raw_text: str,
        top_k: int = 3,
        use_multi_agent: bool = True,
        use_kg: bool = True,
        vector_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        task_id = str(uuid.uuid4())
        text = str(raw_text or "").strip()
        features = self.extractor.extract(text)
        entities = list(features.get("indicators", []))
        query_text = entities_to_query_text(entities) or text

        retrieved_cases = self.case_repo.search(
            query_text,
            k=top_k or settings.default_top_k,
            sources=vector_sources or ["all"],
            top_k_per_source=settings.default_top_k_per_source,
        )
        kg_evidence = (
            self.kg_retriever.retrieve(features=features, raw_text=text, top_k=settings.default_kg_top_k)
            if use_kg
            else []
        )

        if use_multi_agent:
            agent_summary = run_agent_pipeline(entities, retrieved_cases)
            agent_outputs = list(agent_summary.get("agent_outputs", []))
            critique = dict(agent_summary.get("critique", {}))
            overall_risk = str(agent_summary.get("overall_risk", "unknown"))
        else:
            agent_outputs = []
            critique = {}
            overall_risk = "unknown"

        llm_response = self._generate_summary(text, features, retrieved_cases, kg_evidence, agent_outputs, critique)
        prediction = extract_prediction(llm_response)
        possible_diagnoses = self._possible_diagnoses(prediction, retrieved_cases, agent_outputs)
        detection_conclusion = self._detection_conclusion(
            text=text,
            prediction=prediction,
            possible_diagnoses=possible_diagnoses,
            retrieved_cases=retrieved_cases,
            entities=entities,
            overall_risk=overall_risk,
        )
        kg_disease_symptoms = self._kg_disease_symptoms(
            primary_diagnosis=detection_conclusion.get("primary_diagnosis", ""),
            kg_evidence=kg_evidence,
        )
        baseline_match_rates = self._baseline_match_rates(
            text=text,
            prediction=prediction,
            primary_diagnosis=detection_conclusion.get("primary_diagnosis", ""),
            retrieved_cases=retrieved_cases,
            kg_evidence=kg_evidence,
            agent_outputs=agent_outputs,
            entities=entities,
        )
        summary_markdown = self._summary_markdown(
            detection_conclusion=detection_conclusion,
            overall_risk=overall_risk,
            possible_diagnoses=possible_diagnoses,
            baseline_match_rates=baseline_match_rates,
            kg_disease_symptoms=kg_disease_symptoms,
            retrieved_cases=retrieved_cases,
            kg_evidence=kg_evidence,
            agent_outputs=agent_outputs,
            llm_response=llm_response,
        )

        report = FinalReport(
            task_id=task_id,
            overall_risk=overall_risk,
            possible_diagnoses=possible_diagnoses,
            detection_conclusion=detection_conclusion,
            kg_disease_symptoms=kg_disease_symptoms,
            baseline_match_rates=baseline_match_rates,
            retrieved_cases=retrieved_cases,
            kg_evidence=kg_evidence,
            agent_opinions=agent_outputs,
            critique=critique,
            summary_markdown=summary_markdown,
            followup_questions=[
                "Confirm symptom onset, duration, and medication history.",
                "Repeat abnormal lab indicators or consult a qualified clinician when needed.",
            ],
            entities=entities,
            raw_baseline_result={
                "prediction": prediction,
                "llm_response": llm_response,
                "features": features,
                "retrieval_mode": "multi_vector_or_faiss",
                "detection_report_version": "detailed_v1",
            },
        )
        return _dump_model(report)

    def _generate_summary(
        self,
        text: str,
        features: Dict[str, Any],
        retrieved_cases: List[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
        agent_outputs: List[Dict[str, Any]],
        critique: Dict[str, Any],
    ) -> str:
        prompt = (
            "Patient text:\n"
            f"{text}\n\n"
            "Extracted features:\n"
            f"{json.dumps(features, ensure_ascii=False)[:2500]}\n\n"
            "Retrieved cases:\n"
            f"{json.dumps(retrieved_cases[:5], ensure_ascii=False)[:3500]}\n\n"
            "KG evidence:\n"
            f"{json.dumps(kg_evidence[:8], ensure_ascii=False)[:2500]}\n\n"
            "Agent outputs:\n"
            f"{json.dumps(agent_outputs, ensure_ascii=False)[:2000]}\n\n"
            "Critique:\n"
            f"{json.dumps(critique, ensure_ascii=False)[:1500]}\n\n"
            "Give a concise diagnosis-oriented summary for course demonstration."
        )
        return self.llm.generate(
            prompt,
            system_prompt="You are a cautious medical decision-support assistant.",
            mode="API",
        )

    @staticmethod
    def _possible_diagnoses(
        prediction: str,
        retrieved_cases: List[Dict[str, Any]],
        agent_outputs: List[Dict[str, Any]],
    ) -> List[str]:
        labels: List[str] = []
        for row in retrieved_cases:
            diagnosis = str(row.get("diagnosis") or "").strip()
            if diagnosis and diagnosis not in labels:
                labels.append(diagnosis)
            if len(labels) >= 5:
                return labels
        if prediction:
            labels.append(prediction)
        for row in agent_outputs:
            specialty = str(row.get("specialty") or row.get("agent_name") or "").strip()
            risk = str(row.get("risk_level") or "").strip()
            if specialty and risk in {"medium", "high"}:
                labels.append(f"{specialty} risk")
        return labels[:5] or ["general medical risk"]

    @staticmethod
    def _detection_conclusion(
        text: str,
        prediction: str,
        possible_diagnoses: List[str],
        retrieved_cases: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
        overall_risk: str,
    ) -> Dict[str, Any]:
        abnormal_entities = [item for item in entities if item.get("is_abnormal")]
        normal_hint = bool(re.search(r"基本正常|未见明显异常|无明显异常|normal|within normal", text, re.I))
        if normal_hint and not abnormal_entities:
            primary = "未见明确疾病风险"
            conclusion_type = "normal_or_low_risk"
            basis = "OCR/体检文本提示各项指标基本正常，当前规则抽取未发现明显异常指标。"
            confidence = 0.82
        else:
            top_case = retrieved_cases[0] if retrieved_cases else {}
            primary = (
                str(top_case.get("diagnosis") or "").strip()
                or (possible_diagnoses[0] if possible_diagnoses else "")
                or prediction
                or "待进一步临床确认"
            )
            conclusion_type = "suspected_disease"
            abnormal_names = [str(item.get("name")) for item in abnormal_entities if item.get("name")]
            if abnormal_names:
                basis = "发现异常指标：" + "、".join(abnormal_names[:8])
            elif top_case:
                basis = "基于相似病例检索结果给出疑似方向。"
            else:
                basis = "基于输入文本和模型摘要给出初步方向。"
            confidence = _clamp01(float(top_case.get("similarity", 0.55) or 0.55)) if top_case else 0.55
        return {
            "primary_diagnosis": primary,
            "conclusion_type": conclusion_type,
            "overall_risk": overall_risk,
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100, 2),
            "basis": basis,
        }

    @staticmethod
    def _kg_disease_symptoms(
        primary_diagnosis: str,
        kg_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        primary_norm = _norm_text(primary_diagnosis)
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in kg_evidence:
            disease = str(row.get("head") or "").strip()
            if not disease:
                continue
            category = str(row.get("relation_category") or "")
            relation_text = _norm_text(" ".join([str(row.get("relation", "")), str(row.get("tail", ""))]))
            is_symptom = category == "symptom" or any(token in relation_text for token in ["symptom", "cough", "pain", "fever"])
            if primary_norm and primary_norm not in _norm_text(disease) and _norm_text(disease) not in primary_norm:
                if len(grouped) >= 3:
                    continue
            item = grouped.setdefault(
                disease,
                {
                    "disease": disease,
                    "matched_to_primary": bool(primary_norm and (primary_norm in _norm_text(disease) or _norm_text(disease) in primary_norm)),
                    "symptoms": [],
                    "evidence": [],
                },
            )
            if is_symptom:
                symptom = str(row.get("tail") or "").strip()
                if symptom and symptom not in item["symptoms"]:
                    item["symptoms"].append(symptom)
            item["evidence"].append(
                {
                    "head": row.get("head", ""),
                    "relation": row.get("relation", ""),
                    "tail": row.get("tail", ""),
                    "score": row.get("score", 0.0),
                    "relation_category": row.get("relation_category", "general"),
                }
            )
            for neighbor in row.get("neighbors") or []:
                if str(neighbor.get("relation_category") or "") == "symptom":
                    symptom = str(neighbor.get("tail") or "").strip()
                    if symptom and symptom not in item["symptoms"]:
                        item["symptoms"].append(symptom)
        rows = list(grouped.values())
        rows.sort(key=lambda item: (not item.get("matched_to_primary"), -len(item.get("symptoms") or []), -len(item.get("evidence") or [])))
        for item in rows:
            item["symptom_count"] = len(item.get("symptoms") or [])
            item["evidence_count"] = len(item.get("evidence") or [])
        return rows[:5]

    @staticmethod
    def _baseline_match_rates(
        text: str,
        prediction: str,
        primary_diagnosis: str,
        retrieved_cases: List[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
        agent_outputs: List[Dict[str, Any]],
        entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        abnormal_count = len([item for item in entities if item.get("is_abnormal")])
        normal_hint = bool(re.search(r"基本正常|未见明显异常|无明显异常|normal|within normal", text, re.I))
        b0_score = 0.82 if normal_hint and abnormal_count == 0 else min(0.72, 0.45 + abnormal_count * 0.08)
        top_case = retrieved_cases[0] if retrieved_cases else {}
        b1_score = _clamp01(float(top_case.get("similarity", 0.0) or 0.0))
        primary_norm = _norm_text(primary_diagnosis)
        kg_hits = 0
        for row in kg_evidence:
            disease_norm = _norm_text(row.get("head", ""))
            if primary_norm and (primary_norm in disease_norm or disease_norm in primary_norm):
                kg_hits += 1
        kg_score = min(1.0, kg_hits / max(3, min(len(kg_evidence), 8))) if kg_evidence else 0.0
        agent_conf = max([float(row.get("confidence", 0.0) or 0.0) for row in agent_outputs] or [0.0])
        b2_score = _clamp01((b1_score * 0.45) + (kg_score * 0.35) + (agent_conf * 0.20))
        return [
            {
                "mode": "B0",
                "name": "Direct Prompting",
                "prediction": prediction or primary_diagnosis,
                "match_rate": round(b0_score, 4),
                "match_percent": round(b0_score * 100, 2),
                "basis": "仅依据输入文本和异常指标数量估计，不使用检索或知识图谱。",
            },
            {
                "mode": "B1",
                "name": "RAG Similar Case Retrieval",
                "prediction": str(top_case.get("diagnosis") or primary_diagnosis or ""),
                "match_rate": round(b1_score, 4),
                "match_percent": round(b1_score * 100, 2),
                "basis": "取 Top-1 相似病例的向量相似度作为匹配率。",
                "top_case_id": top_case.get("case_id", ""),
            },
            {
                "mode": "B2",
                "name": "KG-RAG + Agent Evidence",
                "prediction": primary_diagnosis,
                "match_rate": round(b2_score, 4),
                "match_percent": round(b2_score * 100, 2),
                "basis": "综合相似病例匹配、KG 疾病证据命中和专科 Agent 置信度。",
                "kg_matched_evidence_count": kg_hits,
                "agent_max_confidence": round(agent_conf, 4),
            },
        ]

    @staticmethod
    def _summary_markdown(
        detection_conclusion: Dict[str, Any],
        overall_risk: str,
        possible_diagnoses: List[str],
        baseline_match_rates: List[Dict[str, Any]],
        kg_disease_symptoms: List[Dict[str, Any]],
        retrieved_cases: List[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
        agent_outputs: List[Dict[str, Any]],
        llm_response: str,
    ) -> str:
        primary = detection_conclusion.get("primary_diagnosis") or "待进一步临床确认"
        conclusion_lines = [
            f"- 初步结论：{primary}",
            f"- 综合风险等级：{overall_risk}",
            f"- 结论置信度：{float(detection_conclusion.get('confidence_percent', 0.0)):.2f}%",
            f"- 判断依据：{detection_conclusion.get('basis', '')}",
        ]
        baseline_lines = []
        for row in baseline_match_rates:
            baseline_lines.append(
                f"- {row.get('mode')}（{row.get('name')}）：{float(row.get('match_percent', 0.0)):.2f}%；"
                f"预测：{row.get('prediction') or 'N/A'}；依据：{row.get('basis')}"
            )
        kg_symptom_lines = []
        for item in kg_disease_symptoms[:5]:
            symptoms = "；".join(str(x) for x in (item.get("symptoms") or [])[:6]) or "未检索到明确症状三元组"
            kg_symptom_lines.append(
                f"- {item.get('disease')}：{symptoms}（证据 {item.get('evidence_count', 0)} 条）"
            )
        case_lines = []
        for idx, case in enumerate(retrieved_cases[:3], start=1):
            label = case.get("diagnosis") or case.get("title") or case.get("case_id")
            score = case.get("similarity", 0.0)
            case_lines.append(f"{idx}. {label}（相似度={float(score):.4f}，case_id={case.get('case_id', '')}）")
        kg_lines = []
        for idx, item in enumerate(kg_evidence[:5], start=1):
            kg_lines.append(
                f"{idx}. {item.get('head', '')} --{item.get('relation', '')}--> {item.get('tail', '')}"
            )
        agent_lines = []
        for item in agent_outputs:
            agent_lines.append(
                f"- {item.get('specialty') or item.get('agent_name')}: "
                f"{item.get('risk_level', 'unknown')} confidence={item.get('confidence', 0)}"
            )
        return "\n".join(
            [
                "## 医疗检测报告",
                "",
                "### 一、检测结论",
                "\n".join(conclusion_lines),
                "",
                "### 二、知识图谱对应疾病症状",
                "\n".join(kg_symptom_lines) if kg_symptom_lines else "未检索到与结论疾病直接对应的症状证据。",
                "",
                "### 三、B0 / B1 / B2 匹配率",
                "\n".join(baseline_lines),
                "",
                "### 四、相似病例证据",
                "\n".join(case_lines) if case_lines else "未检索到相似病例。",
                "",
                "### 五、知识图谱证据明细",
                "\n".join(kg_lines) if kg_lines else "未检索到 KG 证据。",
                "",
                "### 六、专科 Agent 意见",
                "\n".join(agent_lines) if agent_lines else "未启用多 Agent 或未产生专科意见。",
                "",
                "### 七、模型摘要",
                llm_response,
                "",
                "### 八、安全提示",
                "本结果仅用于课程演示和辅助参考，不能替代医生诊断。",
            ]
        )


def _norm_text(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
