# -*- coding: utf-8 -*-
"""Runtime settings for the FastAPI application.

The project intentionally keeps configuration lightweight: values are read from
environment variables and sensible local defaults are used for course demos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


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
    dataset_kg_path: str = os.getenv("DATASET_KG_PATH", "./dataset/knowledge graph of DDXPlus.xlsx")
    data_kg_path: str = os.getenv("DATA_KG_PATH", "./data/kg/knowledge graph of DDXPlus.xlsx")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    report_dir: str = os.getenv("REPORT_DIR", "./data/reports")
    vector_base_dir: str = os.getenv("VECTOR_BASE_DIR", "./vector_db")
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    default_kg_top_k: int = int(os.getenv("DEFAULT_KG_TOP_K", "8"))
    default_top_k_per_source: int = int(os.getenv("DEFAULT_TOP_K_PER_SOURCE", "2"))
    force_mock_llm: bool = _env_bool("FORCE_MOCK_LLM", False)
    use_multi_vector: bool = _env_bool("USE_MULTI_VECTOR", True)

    def preferred_train_dir(self) -> str:
        return self.data_train_dir if _has_json_files(self.data_train_dir) else self.dataset_train_dir

    def preferred_test_dir(self) -> str:
        return self.data_test_dir if _has_json_files(self.data_test_dir) else self.dataset_test_dir

    def preferred_kg_path(self) -> str:
        return self.data_kg_path if os.path.exists(self.data_kg_path) else self.dataset_kg_path


settings = Settings()


def _has_json_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    try:
        return any(name.endswith(".json") for name in os.listdir(path))
    except OSError:
        return False
