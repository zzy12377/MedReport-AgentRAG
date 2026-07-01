# -*- coding: utf-8 -*-
"""Analyze router placeholder."""

def get_router():
    try:
        from fastapi import APIRouter
    except Exception:
        return None
    router = APIRouter(prefix="/analyze", tags=["analyze"])

    @router.get("/ping")
    def ping():
        return {"router": "analyze", "status": "ok"}

    return router

