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
                "确认症状起病时间、持续时间、既往病史和用药史。",
                "对异常指标建议复查，必要时咨询具备资质的临床医生。",
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
            "患者文本：\n"
            f"{text}\n\n"
            "已抽取特征：\n"
            f"{json.dumps(features, ensure_ascii=False)[:2500]}\n\n"
            "相似病例：\n"
            f"{json.dumps(retrieved_cases[:5], ensure_ascii=False)[:3500]}\n\n"
            "知识图谱证据：\n"
            f"{json.dumps(kg_evidence[:8], ensure_ascii=False)[:2500]}\n\n"
            "Agent 输出：\n"
            f"{json.dumps(agent_outputs, ensure_ascii=False)[:2000]}\n\n"
            "冲突检测：\n"
            f"{json.dumps(critique, ensure_ascii=False)[:1500]}\n\n"
            "请只用中文输出一段面向课程演示的诊断摘要，包含：主要风险、关键依据、建议复查或就医方向。"
        )
        return self.llm.generate(
            prompt,
            system_prompt="你是谨慎的医疗辅助诊断助手。必须使用中文回答，不要输出英文段落；医学缩写如 ALT、GLU、LDL-C 可以保留。",
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
        top_case = retrieved_cases[0] if retrieved_cases else {}
        rule_conclusion = _rule_based_conclusion(entities)
        if rule_conclusion:
            primary = rule_conclusion["primary_diagnosis"]
            conclusion_type = rule_conclusion["conclusion_type"]
            basis = rule_conclusion["basis"]
            confidence = rule_conclusion["confidence"]
        elif not abnormal_entities and not top_case:
            primary = "未发现明确异常指标"
            conclusion_type = "no_obvious_abnormality"
            basis = "基于可解析医学指标判断，当前未抽取到明显异常指标；未使用原始报告 conclusion 字段作为判断依据。"
            confidence = 0.65 if entities else 0.35
        else:
            candidate = (
                str(top_case.get("diagnosis") or "").strip()
                or (possible_diagnoses[0] if possible_diagnoses else "")
                or prediction
                or ""
            )
            primary = _clean_prediction_label(candidate) or "待进一步临床确认"
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
        primary_terms = _primary_kg_terms(primary_diagnosis)
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in kg_evidence:
            disease = str(row.get("head") or "").strip()
            if not disease:
                continue
            category = str(row.get("relation_category") or "")
            relation_text = _norm_text(" ".join([str(row.get("relation", "")), str(row.get("tail", ""))]))
            is_symptom = category == "symptom" or any(token in relation_text for token in ["symptom", "cough", "pain", "fever"])
            disease_norm = _norm_text(disease)
            row_blob = _norm_text(
                " ".join(
                    [
                        str(row.get("head", "")),
                        str(row.get("relation", "")),
                        str(row.get("tail", "")),
                        " ".join(str(n.get("tail", "")) for n in (row.get("neighbors") or []) if isinstance(n, dict)),
                    ]
                )
            )
            primary_match = bool(
                primary_norm
                and (
                    primary_norm in disease_norm
                    or disease_norm in primary_norm
                    or any(term in row_blob for term in primary_terms)
                )
            )
            if primary_norm and not primary_match:
                if len(grouped) >= 3:
                    continue
            item = grouped.setdefault(
                disease,
                {
                    "disease": disease,
                    "matched_to_primary": primary_match,
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
        if not rows and primary_terms:
            return [_fallback_kg_symptoms(primary_diagnosis, primary_terms)]
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
        parsed_count = len(entities)
        if abnormal_count > 0:
            b0_score = min(0.72, 0.45 + abnormal_count * 0.08)
        elif parsed_count > 0:
            b0_score = 0.58
        else:
            b0_score = 0.25
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
                "prediction": _clean_prediction_label(prediction) or primary_diagnosis,
                "match_rate": round(b0_score, 4),
                "match_percent": round(b0_score * 100, 2),
                "basis": "仅依据可解析医学指标和异常指标数量估计，不使用原始 conclusion 字段、检索或知识图谱。",
            },
            {
                "mode": "B1",
                "name": "RAG Similar Case Retrieval",
                "prediction": _clean_prediction_label(str(top_case.get("diagnosis") or "")) or primary_diagnosis,
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
            f"- 综合风险等级：{_zh_label(overall_risk)}",
            f"- 结论置信度：{float(detection_conclusion.get('confidence_percent', 0.0)):.2f}%",
            f"- 判断依据：{detection_conclusion.get('basis', '')}",
        ]
        baseline_lines = []
        for row in baseline_match_rates:
            baseline_lines.append(
                f"- {row.get('mode')}（{_zh_label(row.get('name'))}）：{float(row.get('match_percent', 0.0)):.2f}%；"
                f"预测：{_zh_label(row.get('prediction') or 'N/A')}；依据：{_zh_label(row.get('basis'))}"
            )
        kg_symptom_lines = []
        for item in kg_disease_symptoms[:5]:
            symptoms = "；".join(_zh_label(x) for x in (item.get("symptoms") or [])[:6]) or "未检索到明确症状三元组"
            kg_symptom_lines.append(
                f"- {_zh_label(item.get('disease'))}：{symptoms}（证据 {item.get('evidence_count', 0)} 条）"
            )
        case_lines = []
        for idx, case in enumerate(retrieved_cases[:3], start=1):
            label = case.get("diagnosis") or case.get("title") or case.get("case_id")
            score = case.get("similarity", 0.0)
            case_lines.append(f"{idx}. {_zh_label(label)}（相似度={float(score):.4f}，case_id={case.get('case_id', '')}）")
        kg_lines = []
        for idx, item in enumerate(kg_evidence[:5], start=1):
            kg_lines.append(
                f"{idx}. {_zh_label(item.get('head', ''))} --{_zh_label(item.get('relation', ''))}--> {_zh_label(item.get('tail', ''))}"
            )
        agent_lines = []
        for item in agent_outputs:
            agent_lines.append(
                f"- {_zh_label(item.get('specialty') or item.get('agent_name'))}: "
                f"风险等级={_zh_label(item.get('risk_level', 'unknown'))}，置信度={item.get('confidence', 0)}"
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
                _zh_label(llm_response),
                "",
                "### 八、安全提示",
                "本结果仅用于课程演示和辅助参考，不能替代医生诊断。",
            ]
        )


_ZH_EXACT_TERMS = {
    "unknown": "未知",
    "low": "低",
    "medium": "中",
    "high": "高",
    "general medical risk": "一般医学风险",
    "cardiovascular risk": "心血管风险",
    "cardiovascular": "心血管",
    "liver": "肝脏",
    "endocrine": "内分泌",
    "liver function abnormality": "肝功能指标异常",
    "endocrine/metabolic risk": "内分泌或代谢风险",
    "respiratory infection differential": "呼吸道感染相关鉴别",
    "Direct Prompting": "直接提示诊断",
    "RAG Similar Case Retrieval": "RAG 相似病例检索",
    "KG-RAG + Agent Evidence": "知识图谱 RAG 与 Agent 综合证据",
    "N/A": "无",
    "Acute COPD exacerbation / infection": "急性慢阻肺加重或感染",
    "acute_copd_exacerbation_infection": "急性慢阻肺加重或感染",
    "Acute pulmonary edema": "急性肺水肿",
    "acute_pulmonary_edema": "急性肺水肿",
    "Acute rhinosinusitis": "急性鼻窦炎",
    "acute_rhinosinusitis": "急性鼻窦炎",
    "Allergic sinusitis": "过敏性鼻窦炎",
    "allergic_sinusitis": "过敏性鼻窦炎",
    "Anemia": "贫血",
    "anemia": "贫血",
    "Atrial fibrillation": "心房颤动",
    "atrial_fibrillation": "心房颤动",
    "Boerhaave": "Boerhaave 综合征",
    "boerhaave": "Boerhaave 综合征",
    "Bronchiectasis": "支气管扩张",
    "bronchiectasis": "支气管扩张",
    "Bronchitis": "支气管炎",
    "bronchitis": "支气管炎",
    "Bronchospasm / acute asthma exacerbation": "支气管痉挛或急性哮喘加重",
    "bronchospasm_acute_asthma_exacerbation": "支气管痉挛或急性哮喘加重",
    "Chagas": "恰加斯病",
    "chagas": "恰加斯病",
    "Chronic rhinosinusitis": "慢性鼻窦炎",
    "chronic_rhinosinusitis": "慢性鼻窦炎",
    "Cluster headache": "丛集性头痛",
    "cluster_headache": "丛集性头痛",
    "Croup": "哮吼",
    "croup": "哮吼",
    "Ebola": "埃博拉病毒病",
    "ebola": "埃博拉病毒病",
    "Epiglottitis": "会厌炎",
    "epiglottitis": "会厌炎",
    "GERD": "胃食管反流病",
    "gerd": "胃食管反流病",
    "Guillain-Barré syndrome": "吉兰-巴雷综合征",
    "guillain_barre_syndrome": "吉兰-巴雷综合征",
    "HIV (initial infection)": "HIV 初次感染",
    "hiv_initial_infection": "HIV 初次感染",
    "Influenza": "流行性感冒",
    "influenza": "流行性感冒",
    "Inguinal hernia": "腹股沟疝",
    "inguinal_hernia": "腹股沟疝",
    "Larygospasm": "喉痉挛",
    "larygospasm": "喉痉挛",
    "Localized edema": "局部水肿",
    "localized_edema": "局部水肿",
    "Myasthenia gravis": "重症肌无力",
    "myasthenia_gravis": "重症肌无力",
    "Myocarditis": "心肌炎",
    "myocarditis": "心肌炎",
    "Pancreatic neoplasm": "胰腺肿瘤",
    "pancreatic_neoplasm": "胰腺肿瘤",
    "Panic attack": "惊恐发作",
    "panic_attack": "惊恐发作",
    "Pericarditis": "心包炎",
    "pericarditis": "心包炎",
    "Pneumonia": "肺炎",
    "pneumonia": "肺炎",
    "Possible NSTEMI / STEMI": "疑似非 ST 段抬高型或 ST 段抬高型心肌梗死",
    "possible_nstemi_stemi": "疑似非 ST 段抬高型或 ST 段抬高型心肌梗死",
    "PSVT": "阵发性室上性心动过速",
    "psvt": "阵发性室上性心动过速",
    "Pulmonary embolism": "肺栓塞",
    "pulmonary_embolism": "肺栓塞",
    "Pulmonary neoplasm": "肺部肿瘤",
    "pulmonary_neoplasm": "肺部肿瘤",
    "Sarcoidosis": "结节病",
    "sarcoidosis": "结节病",
    "Scombroid food poisoning": "鲭鱼中毒",
    "scombroid_food_poisoning": "鲭鱼中毒",
    "SLE": "系统性红斑狼疮",
    "sle": "系统性红斑狼疮",
    "Spontaneous pneumothorax": "自发性气胸",
    "spontaneous_pneumothorax": "自发性气胸",
    "Spontaneous rib fracture": "自发性肋骨骨折",
    "spontaneous_rib_fracture": "自发性肋骨骨折",
    "Stable angina": "稳定型心绞痛",
    "stable_angina": "稳定型心绞痛",
    "Tuberculosis": "结核病",
    "tuberculosis": "结核病",
    "URTI": "上呼吸道感染",
    "urti": "上呼吸道感染",
    "Unstable angina": "不稳定型心绞痛",
    "unstable_angina": "不稳定型心绞痛",
    "Viral pharyngitis": "病毒性咽炎",
    "viral_pharyngitis": "病毒性咽炎",
    "Whooping cough": "百日咳",
    "whooping_cough": "百日咳",
}


_ZH_REPLACE_TERMS = [
    ("Mock API diagnosis", "模拟 API 诊断"),
    ("Mock B0 diagnosis", "模拟 B0 诊断"),
    ("Mock B1 diagnosis", "模拟 B1 诊断"),
    ("Mock B2 diagnosis", "模拟 B2 诊断"),
    ("This mock output is used because no available LLM call was completed.", "当前使用本地模拟输出，因为没有完成可用的大模型调用。"),
    ("cardiovascular risk", "心血管风险"),
    ("liver function abnormality", "肝功能指标异常"),
    ("endocrine/metabolic risk", "内分泌或代谢风险"),
    ("respiratory infection differential", "呼吸道感染相关鉴别"),
    ("general medical risk", "一般医学风险"),
    ("Stable angina", "稳定型心绞痛"),
    ("Unstable angina", "不稳定型心绞痛"),
    ("Possible NSTEMI / STEMI", "疑似非 ST 段抬高型或 ST 段抬高型心肌梗死"),
    ("Myocarditis", "心肌炎"),
    ("Atrial fibrillation", "心房颤动"),
    ("Panic attack", "惊恐发作"),
    ("Anemia", "贫血"),
    ("GERD", "胃食管反流病"),
    ("Localized edema", "局部水肿"),
    ("Pulmonary embolism", "肺栓塞"),
    ("Anaphylaxis", "过敏性休克"),
    ("Tuberculosis", "结核病"),
    ("tuberculosis", "结核病"),
    ("urti", "上呼吸道感染"),
    ("acute_rhinosinusitis", "急性鼻窦炎"),
    ("hypertension", "高血压"),
    ("blood pressure", "血压"),
    ("heart disease", "心脏疾病"),
    ("cholesterol", "胆固醇"),
    ("atherosclerosis", "动脉粥样硬化"),
    ("diabetes", "糖尿病"),
    ("glucose", "血糖"),
    ("hyperglycemia", "高血糖"),
    ("liver", "肝脏"),
    ("hepatic", "肝脏相关"),
    ("hepatitis", "肝炎"),
    ("has symptomatology", "症状表现为"),
    ("has anamnesis", "病史提示"),
    ("has risk factor", "危险因素为"),
    ("has exposure", "暴露史"),
    ("has lifestyle", "生活方式相关"),
    ("has therapy", "治疗方式包括"),
    ("has treatment", "治疗方式包括"),
    ("has complication", "并发症包括"),
    ("has biological", "生物学相关"),
    ("has differential diagnosis", "鉴别诊断包括"),
    ("Have you traveled out of the country in the last 4 weeks?: N", "过去 4 周内是否出国旅行：否"),
    ("Have you traveled out of the country in the last 4 weeks?: Y", "过去 4 周内是否出国旅行：是"),
    ("Have you traveled out of the country in the last 4 weeks?", "过去 4 周内是否出国旅行"),
    ("Worsening shortness of breath, chronic cough with sputum production", "气短加重，慢性咳嗽伴咳痰"),
    ("Increased sputum purulence and volume", "痰液脓性和痰量增加"),
    ("Common in smokers or individuals exposed to pollutants", "常见于吸烟者或污染物暴露人群"),
    ("History of chronic obstructive pulmonary disease (COPD)", "慢性阻塞性肺疾病病史（COPD）"),
    ("Bronchodilators and antibiotics if bacterial infection is present", "如存在细菌感染，可使用支气管扩张剂和抗生素"),
    ("Persistent cough lasting more than 3 weeks", "持续超过 3 周的咳嗽"),
    ("night sweats", "夜间盗汗"),
    ("weight loss", "体重下降"),
    ("Sore throat", "咽痛"),
    ("nasal congestion", "鼻塞"),
    ("cough", "咳嗽"),
    ("fever", "发热"),
    ("pain", "疼痛"),
    ("headache", "头痛"),
    ("dizziness", "头晕"),
    ("shortness of breath", "气短"),
    ("fatigue", "乏力"),
]

for _src_key, _dst_value in list(_ZH_EXACT_TERMS.items()):
    _ZH_EXACT_TERMS.setdefault(_src_key.lower(), _dst_value)
    _ZH_EXACT_TERMS.setdefault(re.sub(r"[^a-z0-9]+", "_", _src_key.lower()).strip("_"), _dst_value)


def _zh_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    norm_key = _zh_lookup_key(text)
    exact = (
        _ZH_EXACT_TERMS.get(text)
        or _ZH_EXACT_TERMS.get(text.lower())
        or _ZH_EXACT_TERMS.get(norm_key)
    )
    if exact:
        return exact
    translated = text
    for src, dst in _ZH_REPLACE_TERMS:
        translated = re.sub(re.escape(src), dst, translated, flags=re.IGNORECASE)
    translated = _translate_underscore_terms(translated)
    translated = translated.replace("confidence=", "置信度=")
    translated = translated.replace("Diagnosis:", "诊断：")
    translated = translated.replace("prediction:", "预测：")
    translated = translated.replace(": N", "：否").replace(": Y", "：是")
    return translated


def _zh_lookup_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text or "").strip().lower()).strip("_")


def _translate_underscore_terms(text: str) -> str:
    result = str(text or "")
    for token in sorted(re.findall(r"\b[a-z][a-z0-9_]{2,}\b", result), key=len, reverse=True):
        zh = _ZH_EXACT_TERMS.get(token) or _ZH_EXACT_TERMS.get(_zh_lookup_key(token))
        if zh:
            result = re.sub(rf"\b{re.escape(token)}\b", zh, result)
    return result


def _norm_text(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _rule_based_conclusion(entities: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    abnormal = [row for row in entities if row.get("is_abnormal")]
    if not abnormal:
        return None
    names = {_norm_text(row.get("name", "")) for row in abnormal}
    values = {str(row.get("name", "")): row.get("value") for row in abnormal}
    if names & {_norm_text("收缩压"), _norm_text("舒张压"), "sbp", "dbp"}:
        sbp = _value_for_entity(abnormal, {"收缩压", "SBP"})
        dbp = _value_for_entity(abnormal, {"舒张压", "DBP"})
        bp_text = []
        if sbp is not None:
            bp_text.append(f"收缩压 {sbp:g} mmHg")
        if dbp is not None:
            bp_text.append(f"舒张压 {dbp:g} mmHg")
        return {
            "primary_diagnosis": "血压升高 / 高血压风险",
            "conclusion_type": "vital_sign_abnormality",
            "basis": "发现异常血压指标：" + ("、".join(bp_text) if bp_text else "血压超过参考范围"),
            "confidence": 0.86 if sbp is not None and dbp is not None else 0.78,
        }
    if names & {"glu", "hba1c"}:
        return {
            "primary_diagnosis": "血糖异常 / 糖代谢风险",
            "conclusion_type": "metabolic_abnormality",
            "basis": "发现血糖或糖化血红蛋白异常：" + "、".join(str(row.get("name")) for row in abnormal[:6]),
            "confidence": 0.8,
        }
    if names & {"tc", "tg", "ldl c", "ldl", "hdl c", "hdl"}:
        return {
            "primary_diagnosis": "血脂异常 / 心血管代谢风险",
            "conclusion_type": "metabolic_abnormality",
            "basis": "发现血脂相关指标异常：" + "、".join(str(row.get("name")) for row in abnormal[:6]),
            "confidence": 0.78,
        }
    if names & {"alt", "ast", "ggt", "tbil", "alp"}:
        return {
            "primary_diagnosis": "肝功能指标异常",
            "conclusion_type": "lab_abnormality",
            "basis": "发现肝功能相关指标异常：" + "、".join(str(row.get("name")) for row in abnormal[:6]),
            "confidence": 0.76,
        }
    return {
        "primary_diagnosis": "体检指标异常，需进一步评估",
        "conclusion_type": "general_abnormality",
        "basis": "发现异常指标：" + "、".join(str(row.get("name")) for row in abnormal[:8]),
        "confidence": 0.68,
    }


def _value_for_entity(entities: List[Dict[str, Any]], target_names: set[str]) -> float | None:
    target_norms = {_norm_text(name) for name in target_names}
    for row in entities:
        if _norm_text(row.get("name", "")) in target_norms:
            try:
                return float(row.get("value"))
            except Exception:
                return None
    return None


def _clean_prediction_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"(?i)^diagnosis[- ]oriented summary:\s*", "", text).strip()
    first_line = re.split(r"[\n。；;]", text, maxsplit=1)[0].strip()
    first_line = re.sub(r"(?i)^the\s+\d{1,3}[- ]year[- ]old.*?presents with\s+", "", first_line).strip()
    if len(first_line) > 80:
        bp_match = re.search(r"\bhypertension\b|blood pressure|systolic|diastolic", first_line, re.I)
        if bp_match:
            return "血压升高 / 高血压风险"
        return first_line[:80].rstrip(" ,.;:")
    if re.search(r"\bhypertension\b|blood pressure|systolic|diastolic", first_line, re.I):
        return "血压升高 / 高血压风险"
    return first_line


def _primary_kg_terms(primary_diagnosis: str) -> List[str]:
    norm = _norm_text(primary_diagnosis)
    if any(term in norm for term in ["高血压", "血压", "hypertension"]):
        return ["hypertension", "blood pressure", "heart disease", "cardiovascular"]
    if any(term in norm for term in ["血糖", "糖", "diabetes", "glucose"]):
        return ["diabetes", "glucose", "hyperglycemia"]
    if any(term in norm for term in ["血脂", "心血管", "cholesterol", "lipid"]):
        return ["cholesterol", "atherosclerosis", "cardiovascular", "heart disease"]
    if any(term in norm for term in ["肝", "alt", "ast", "liver"]):
        return ["liver", "hepatic", "hepatitis"]
    return []


def _fallback_kg_symptoms(primary_diagnosis: str, primary_terms: List[str]) -> Dict[str, Any]:
    if "hypertension" in primary_terms or "blood pressure" in primary_terms:
        symptoms = ["多数早期可无明显症状", "可出现头痛、头晕、心悸", "长期控制不佳可增加心脑血管风险"]
    elif "diabetes" in primary_terms:
        symptoms = ["多饮、多尿、多食", "乏力或体重变化", "部分早期可无明显症状"]
    elif "liver" in primary_terms:
        symptoms = ["乏力、食欲下降", "右上腹不适", "严重时可出现黄疸"]
    else:
        symptoms = ["未检索到明确症状三元组"]
    return {
        "disease": primary_diagnosis,
        "matched_to_primary": True,
        "symptoms": symptoms,
        "evidence": [],
        "symptom_count": len(symptoms),
        "evidence_count": 0,
        "note": "KG 未返回直接匹配三元组，使用内置医学规则补充展示；正式结论仍以指标、检索和 KG 证据综合为准。",
    }
