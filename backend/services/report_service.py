# -*- coding: utf-8 -*-
"""Report persistence helpers."""

from __future__ import annotations

from baselines.common import save_result


def save_report(result: dict, output_dir: str = "./storage/results") -> dict:
    return save_result(result, output_dir)

