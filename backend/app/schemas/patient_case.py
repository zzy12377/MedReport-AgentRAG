# -*- coding: utf-8 -*-
"""Patient input and extracted-feature schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PatientIndicator(BaseModel):
    name: str = Field(..., description="Normalized indicator name, e.g. ALT, GLU, LDL-C")
    value: Optional[float] = None
    unit: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    reference_range: Optional[str] = None
    is_abnormal: Optional[bool] = None
    original_text: str = ""


class PatientFeatures(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None
    symptoms: str = ""
    location: str = ""
    restriction: str = ""
    indicators: List[PatientIndicator] = Field(default_factory=list)


class DiagnosisRequest(BaseModel):
    text: str = Field(..., description="User text or OCR-recognized report text")
    top_k: int = Field(3, ge=1, le=30)
    use_multi_agent: bool = True
    use_kg: bool = True
    vector_sources: Optional[List[str]] = Field(
        default=None,
        description="Optional vector store names, e.g. ['all'] or ['ddxplus_cases']",
    )


class OcrJsonDiagnosisRequest(BaseModel):
    ocr_json: Dict[str, Any] = Field(..., description="OCR-recognized JSON payload")
    case_id: Optional[str] = Field(default=None, description="Optional frontend case id")
    top_k: int = Field(3, ge=1, le=30)
    use_multi_agent: bool = True
    use_kg: bool = True
    vector_sources: Optional[List[str]] = Field(
        default=None,
        description="Optional vector store names, e.g. ['all'] or ['ddxplus_cases']",
    )


class DiagnosisResponse(BaseModel):
    task_id: str
    status: str
    report: dict
