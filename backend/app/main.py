# -*- coding: utf-8 -*-
"""FastAPI entrypoint required by the course project documents."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.router_chat import router as chat_router
from backend.app.api.router_diagnosis import router as diagnosis_router
from backend.app.api.router_history import router as history_router
from backend.app.api.router_task import router as task_router
from backend.app.api.router_upload import router as upload_router
from backend.app.config.settings import settings
from backend.app.core.kg.kg_repository import KGRepository
from backend.app.core.retrieval.case_repository import CaseRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("[STARTUP] Loading case repository...")
        CaseRepository.get_instance().load()
    except Exception as exc:
        print(f"[WARN] Case repository startup load skipped: {exc}")
    try:
        print("[STARTUP] Loading knowledge graph...")
        KGRepository.get_instance().load()
    except Exception as exc:
        print(f"[WARN] KG startup load skipped: {exc}")
    print("[STARTUP] Backend ready.")
    yield
    print("[SHUTDOWN] Backend stopped.")


app = FastAPI(
    title=settings.app_name,
    description="Multimodal medical report interpretation and multi-agent assisted diagnosis system",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(diagnosis_router, prefix=settings.api_prefix, tags=["diagnosis"])
app.include_router(task_router, prefix=settings.api_prefix, tags=["task"])
app.include_router(upload_router, prefix=settings.api_prefix, tags=["upload"])
app.include_router(history_router, prefix=settings.api_prefix, tags=["history"])
app.include_router(chat_router, prefix=settings.api_prefix, tags=["chat"])


@app.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "status": "running",
        "api_prefix": settings.api_prefix,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "backend.app.main"}
