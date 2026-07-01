# -*- coding: utf-8 -*-
"""Upload router placeholder."""

def get_router():
    try:
        from fastapi import APIRouter
    except Exception:
        return None
    router = APIRouter(prefix="/upload", tags=["upload"])

    @router.get("/ping")
    def ping():
        return {"router": "upload", "status": "ok"}

    return router

