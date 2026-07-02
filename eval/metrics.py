# -*- coding: utf-8 -*-
"""Compatibility wrapper for the DDXPlus metrics implementation."""

from __future__ import annotations

from metrics.metrics_DDXPlus import *  # noqa: F401,F403
from metrics.metrics_DDXPlus import main


if __name__ == "__main__":
    raise SystemExit(main())
