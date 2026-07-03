# -*- coding: utf-8 -*-
"""Rule-based specialist agents for physical-exam report interpretation."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


SPECIALTIES: Dict[str, Dict[str, Any]] = {
    "cardiovascular": {
        "signals": {"TC", "TG", "LDL-C", "HDL-C", "收缩压", "舒张压", "BMI", "心率"},
        "diagnoses": {
            "收缩压": "高血压风险",
            "舒张压": "高血压风险",
            "LDL-C": "血脂异常/动脉粥样硬化风险",
            "HDL-C": "血脂异常风险",
            "TC": "血脂异常风险",
            "TG": "血脂异常风险",
            "BMI": "心血管代谢风险",
            "心率": "心率异常需结合症状评估",
        },
        "suggestion": "建议复测血压和血脂，评估生活方式、吸烟史、家族史；高危者咨询心内科。",
    },
    "liver": {
        "signals": {"ALT", "AST", "GGT", "ALP", "TBIL", "DBIL", "IBIL", "ALB"},
        "diagnoses": {
            "ALT": "肝细胞损伤风险",
            "AST": "肝细胞损伤风险",
            "GGT": "胆汁淤积或脂肪肝相关风险",
            "ALP": "胆道/骨代谢相关指标异常",
            "TBIL": "胆红素异常风险",
            "DBIL": "胆红素异常风险",
            "IBIL": "胆红素异常风险",
            "ALB": "肝合成功能或营养状态风险",
        },
        "suggestion": "建议结合饮酒、用药、乙肝/丙肝筛查和腹部超声，必要时复查肝功能。",
    },
    "endocrine": {
        "signals": {"GLU", "HbA1c", "尿酸", "TSH", "FT3", "FT4", "BMI", "BUN", "Cr"},
        "diagnoses": {
            "GLU": "糖尿病或糖代谢异常风险",
            "HbA1c": "糖尿病或长期血糖控制异常风险",
            "尿酸": "高尿酸血症/痛风风险",
            "TSH": "甲状腺功能异常风险",
            "FT3": "甲状腺功能异常风险",
            "FT4": "甲状腺功能异常风险",
            "BMI": "肥胖相关代谢风险",
            "BUN": "肾功能/代谢异常风险",
            "Cr": "肾功能异常风险",
        },
        "suggestion": "建议复查空腹血糖、HbA1c、尿酸和肾功能，并结合饮食、体重和内分泌专科评估。",
    },
    "hematology": {
        "signals": {"WBC", "RBC", "HGB", "PLT"},
        "diagnoses": {
            "WBC": "感染/炎症或血液系统异常风险",
            "RBC": "红细胞异常风险",
            "HGB": "贫血风险",
            "PLT": "血小板异常风险",
        },
        "suggestion": "建议结合发热、出血、感染症状和血常规复查，必要时咨询血液科。",
    },
}


class SpecialistAgent:
    def __init__(self, specialty: str):
        self.specialty = specialty
        self.config = SPECIALTIES.get(specialty, {})
        self.signals = self.config.get("signals", set())

    def analyze(
        self,
        entities: Iterable[Dict[str, object]],
        retrieved_cases: List[Dict[str, object]] | None = None,
        kg_evidence: List[Dict[str, object]] | None = None,
    ) -> Dict[str, object]:
        matched = [e for e in entities if e.get("name") in self.signals]
        abnormal = [e for e in matched if e.get("is_abnormal")]
        kg_hits = self._kg_hits(kg_evidence or [])
        risk = self._risk_level(abnormal, kg_hits)
        confidence = self._confidence(matched, abnormal, kg_hits, retrieved_cases or [])
        diagnoses = self._diagnoses(abnormal)
        evidence = [
            f"{e.get('name')}={e.get('value')} {e.get('unit')} "
            f"(ref {e.get('ref_low')}-{e.get('ref_high')}, abnormal={e.get('is_abnormal')})"
            for e in matched
        ]
        evidence.extend([str(row.get("text") or f"{row.get('head')}->{row.get('tail')}") for row in kg_hits[:3]])
        return {
            "specialty": self.specialty,
            "agent_name": self.specialty,
            "risk_level": risk,
            "confidence": confidence,
            "diagnosis": diagnoses,
            "evidence": evidence,
            "matched_indicators": matched,
            "kg_evidence_count": len(kg_hits),
            "retrieved_case_count": len(retrieved_cases or []),
            "suggestion": self.config.get("suggestion", ""),
        }

    def _diagnoses(self, abnormal: List[Dict[str, object]]) -> List[str]:
        mapping = self.config.get("diagnoses", {})
        labels = []
        for entity in abnormal:
            label = mapping.get(entity.get("name"))
            if label and label not in labels:
                labels.append(label)
        return labels or ["未见明确专科高风险信号"]

    def _kg_hits(self, kg_evidence: List[Dict[str, object]]) -> List[Dict[str, object]]:
        rows = []
        for row in kg_evidence:
            text = " ".join(str(row.get(k, "")) for k in ["head", "tail", "indicator", "text"])
            if any(signal in text for signal in self.signals):
                rows.append(row)
        return rows

    @staticmethod
    def _risk_level(abnormal: List[Dict[str, object]], kg_hits: List[Dict[str, object]]) -> str:
        score = len(abnormal) + min(len(kg_hits), 3) * 0.5
        if score >= 3:
            return "high"
        if score >= 1:
            return "medium"
        return "low"

    @staticmethod
    def _confidence(
        matched: List[Dict[str, object]],
        abnormal: List[Dict[str, object]],
        kg_hits: List[Dict[str, object]],
        retrieved_cases: List[Dict[str, object]],
    ) -> float:
        score = 0.35
        score += min(len(matched), 4) * 0.07
        score += min(len(abnormal), 4) * 0.08
        score += min(len(kg_hits), 3) * 0.05
        score += 0.05 if retrieved_cases else 0.0
        return round(min(score, 0.95), 2)
