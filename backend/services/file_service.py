# -*- coding: utf-8 -*-
"""File service placeholder."""

from __future__ import annotations

import os


def ensure_storage_dirs() -> None:
    for path in ["./storage/uploads", "./storage/results", "./storage/knowledge"]:
        os.makedirs(path, exist_ok=True)

