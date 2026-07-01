# -*- coding: utf-8 -*-
"""Chat router placeholder."""

def get_router():
    try:
        from fastapi import APIRouter
    except Exception:
        return None
    router = APIRouter(prefix="/chat", tags=["chat"])

    @router.get("/ping")
    def ping():
        return {"router": "chat", "status": "ok"}

    return router

