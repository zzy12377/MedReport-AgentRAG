# -*- coding: utf-8 -*-
"""History router placeholder."""

def get_router():
    try:
        from fastapi import APIRouter
    except Exception:
        return None
    router = APIRouter(prefix="/history", tags=["history"])

    @router.get("/ping")
    def ping():
        return {"router": "history", "status": "ok"}

    return router

