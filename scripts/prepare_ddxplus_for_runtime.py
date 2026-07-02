# -*- coding: utf-8 -*-
"""Runtime-name wrapper for the existing DDXPlus preparation script."""

from __future__ import annotations

import runpy


if __name__ == "__main__":
    runpy.run_module("scripts.prepare_ddxplus_for_medrag", run_name="__main__")
