# -*- coding: utf-8 -*-
"""Pipeline orchestration for baseline modes."""

from __future__ import annotations

from baselines.run_b0_direct import run_b0
from baselines.run_b1_rag import run_b1
from baselines.run_b2_kg_rag import run_b2


def run_pipeline(text: str, mode: str = "B1", mock: bool = False) -> dict:
    mode = (mode or "B1").upper()
    if mode == "B0":
        return run_b0(text, mock=mock)
    if mode == "B2":
        return run_b2(text, mock=True)
    return run_b1(text, mock=mock)

