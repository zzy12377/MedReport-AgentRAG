# -*- coding: utf-8 -*-
"""Real-time diagnosis pipeline used by the FastAPI interface."""

from __future__ import annotations

import json
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
            agent_summary = run_agent_pipeline(entities, retrieved_cases, kg_evidence)
            agent_outputs = list(agent_summary.get("agent_outputs", []))
            critique = dict(agent_summary.get("critique", {}))
            overall_risk = str(agent_summary.get("overall_risk", "unknown"))
            recommendations = list(agent_summary.get("recommendations", []))
        else:
            agent_outputs = []
            critique = {}
            overall_risk = "unknown"
            recommendations = []

        llm_response = self._generate_summary(text, features, retrieved_cases, kg_evidence, agent_outputs, critique)
        prediction = extract_prediction(llm_response)
        possible_diagnoses = self._possible_diagnoses(prediction, retrieved_cases, agent_outputs)
        summary_markdown = self._summary_markdown(
            prediction=prediction,
            overall_risk=overall_risk,
            retrieved_cases=retrieved_cases,
            kg_evidence=kg_evidence,
            agent_outputs=agent_outputs,
            llm_response=llm_response,
        )

        report = FinalReport(
            task_id=task_id,
            overall_risk=overall_risk,
            possible_diagnoses=possible_diagnoses,
            retrieved_cases=retrieved_cases,
            kg_evidence=kg_evidence,
            agent_opinions=agent_outputs,
            critique=critique,
            summary_markdown=summary_markdown,
            followup_questions=[
                "Confirm symptom onset, duration, and medication history.",
                "Repeat abnormal lab indicators or consult a qualified clinician when needed.",
            ]
            + recommendations[:3],
            entities=entities,
            raw_baseline_result={
                "prediction": prediction,
                "llm_response": llm_response,
                "features": features,
                "retrieval_mode": "multi_vector_or_faiss",
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
    def _summary_markdown(
        prediction: str,
        overall_risk: str,
        retrieved_cases: List[Dict[str, Any]],
        kg_evidence: List[Dict[str, Any]],
        agent_outputs: List[Dict[str, Any]],
        llm_response: str,
    ) -> str:
        case_lines = []
        for idx, case in enumerate(retrieved_cases[:3], start=1):
            label = case.get("diagnosis") or case.get("title") or case.get("case_id")
            score = case.get("similarity", 0.0)
            case_lines.append(f"{idx}. {label} (score={float(score):.4f})")
        kg_lines = []
        for idx, item in enumerate(kg_evidence[:5], start=1):
            kg_lines.append(
                f"{idx}. {item.get('head', '')} --{item.get('relation', '')}--> {item.get('tail', '')}"
            )
        agent_lines = []
        for item in agent_outputs:
            agent_lines.append(
                f"- {item.get('specialty') or item.get('agent_name')}: "
                f"{item.get('risk_level', 'unknown')} confidence={item.get('confidence', 0)}; "
                f"diagnosis={', '.join(item.get('diagnosis') or [])}"
            )
        return "\n".join(
            [
                "## Diagnosis Report",
                "",
                f"**Overall risk:** {overall_risk}",
                f"**Prediction:** {prediction or 'N/A'}",
                "",
                "### Similar Cases",
                "\n".join(case_lines) if case_lines else "No similar cases retrieved.",
                "",
                "### KG Evidence",
                "\n".join(kg_lines) if kg_lines else "No KG evidence retrieved.",
                "",
                "### Agent Opinions",
                "\n".join(agent_lines) if agent_lines else "Multi-agent analysis was disabled or produced no opinion.",
                "",
                "### LLM Summary",
                llm_response,
                "",
                "> This result is for course demonstration and reference only; it cannot replace a physician diagnosis.",
            ]
        )
