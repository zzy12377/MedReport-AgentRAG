# -*- coding: utf-8 -*-
"""Runtime settings for the FastAPI application.

The project intentionally keeps configuration lightweight: values are read from
environment variables and sensible local defaults are used for course demos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
    load_dotenv()
except Exception:
    def _load_env_file(path: Path) -> None:
        if not path.exists():
            return
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)

    for _candidate in (_PROJECT_ROOT / ".env", Path.cwd() / ".env"):
        _load_env_file(_candidate)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "MedReport AgentRAG")
    api_prefix: str = os.getenv("API_PREFIX", "/api/v1")
    dataset_train_dir: str = os.getenv("DATASET_TRAIN_DIR", "./dataset/df/train")
    dataset_test_dir: str = os.getenv("DATASET_TEST_DIR", "./dataset/df/test")
    data_train_dir: str = os.getenv("DATA_TRAIN_DIR", "./data/df/train")
    data_test_dir: str = os.getenv("DATA_TEST_DIR", "./data/df/test")
    data_zh_train_dir: str = os.getenv("DATA_ZH_TRAIN_DIR", "./data_zh/df/train")
    data_zh_test_dir: str = os.getenv("DATA_ZH_TEST_DIR", "./data_zh/df/test")
    dataset_kg_path: str = os.getenv("DATASET_KG_PATH", "./data_zh/kg/combined_kg_zh.jsonl")
    data_kg_path: str = os.getenv("DATA_KG_PATH", "./data_zh/kg/combined_kg_zh.jsonl")
    data_zh_kg_path: str = os.getenv("DATA_ZH_KG_PATH", "./data_zh/kg/ddxplus_kg_zh.jsonl")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    report_dir: str = os.getenv("REPORT_DIR", "./data/reports")
    vector_base_dir: str = os.getenv("VECTOR_BASE_DIR", "./vector_db")
    external_zh_data_dir: str = os.getenv("EXTERNAL_ZH_DATA_DIR", "./data_zh/external")
    external_vector_base_dir: str = os.getenv("EXTERNAL_VECTOR_BASE_DIR", "./vector_db_zh")
    external_kg_path: str = os.getenv("EXTERNAL_KG_PATH", "./data_zh/kg/external_case_evidence_zh.jsonl")
    combined_zh_kg_path: str = os.getenv("COMBINED_ZH_KG_PATH", "./data_zh/kg/combined_kg_zh.jsonl")
    case_index_path: str = os.getenv("CASE_INDEX_PATH", "./storage/indexes/ddxplus_cases.faiss")
    case_index_metadata_path: str = os.getenv("CASE_INDEX_METADATA_PATH", "./storage/indexes/ddxplus_cases_metadata.jsonl")
    case_embeddings_path: str = os.getenv("CASE_EMBEDDINGS_PATH", "./storage/embeddings/ddxplus_cases.npy")
    case_embedding_metadata_path: str = os.getenv("CASE_EMBEDDING_METADATA_PATH", "./storage/embeddings/ddxplus_cases_metadata.jsonl")
    case_zh_index_path: str = os.getenv("CASE_ZH_INDEX_PATH", "./storage/indexes/ddxplus_cases_zh.faiss")
    case_zh_index_metadata_path: str = os.getenv("CASE_ZH_INDEX_METADATA_PATH", "./storage/indexes/ddxplus_cases_zh_metadata.jsonl")
    case_zh_embeddings_path: str = os.getenv("CASE_ZH_EMBEDDINGS_PATH", "./storage/embeddings/ddxplus_cases_zh.npy")
    case_zh_embedding_metadata_path: str = os.getenv("CASE_ZH_EMBEDDING_METADATA_PATH", "./storage/embeddings/ddxplus_cases_zh_metadata.jsonl")
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    default_kg_top_k: int = int(os.getenv("DEFAULT_KG_TOP_K", "8"))
    default_top_k_per_source: int = int(os.getenv("DEFAULT_TOP_K_PER_SOURCE", "2"))
    force_mock_llm: bool = _env_bool("FORCE_MOCK_LLM", False)
    use_multi_vector: bool = _env_bool("USE_MULTI_VECTOR", True)
    use_zh_data: bool = _env_bool("USE_ZH_DATA", False)
    use_external_vector: bool = _env_bool("USE_EXTERNAL_VECTOR", False)
    use_llm_mdt_agent: bool = _env_bool("USE_LLM_MDT_AGENT", False)
    llm_mdt_max_cases: int = int(os.getenv("LLM_MDT_MAX_CASES", "3"))
    llm_mdt_max_kg: int = int(os.getenv("LLM_MDT_MAX_KG", "5"))
    external_vector_sources: str = os.getenv(
        "EXTERNAL_VECTOR_SOURCES",
        "medcase_reasoning,open_patients,pmc_patients",
    )

    def preferred_train_dir(self) -> str:
        if self.use_zh_data and _has_json_files(self.data_zh_train_dir):
            return self.data_zh_train_dir
        return self.data_train_dir if _has_json_files(self.data_train_dir) else self.dataset_train_dir

    def preferred_test_dir(self) -> str:
        if self.use_zh_data and _has_json_files(self.data_zh_test_dir):
            return self.data_zh_test_dir
        return self.data_test_dir if _has_json_files(self.data_test_dir) else self.dataset_test_dir

    def preferred_kg_path(self) -> str:
        if self.use_zh_data and os.path.exists(self.combined_zh_kg_path):
            return self.combined_zh_kg_path
        if self.use_zh_data and os.path.exists(self.data_zh_kg_path):
            return self.data_zh_kg_path
        return self.data_kg_path if os.path.exists(self.data_kg_path) else self.dataset_kg_path

    def preferred_external_vector_sources(self) -> list[str]:
        value = str(self.external_vector_sources or "").strip()
        if not value or value.lower() == "all":
            return ["all"]
        return [item.strip() for item in value.split(",") if item.strip()]


settings = Settings()


def _has_json_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        return any(name.endswith(".json") for name in os.listdir(path))
    except OSError:
        return False
