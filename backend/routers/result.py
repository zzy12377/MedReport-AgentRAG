# -*- coding: utf-8 -*-
"""Result router placeholder."""

def get_router():
    try:
        from fastapi import APIRouter
    except Exception:
        return None
    router = APIRouter(prefix="/result", tags=["result"])

    @router.get("/ping")
    def ping():
        return {"router": "result", "status": "ok"}

    return router

