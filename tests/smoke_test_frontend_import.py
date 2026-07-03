# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> int:
    from frontend import app_gradio
    from frontend.components import chat_tab, history_tab, result_tab, upload_tab

    assert app_gradio.API_BASE
    for module in [chat_tab, history_tab, result_tab, upload_tab]:
        if not hasattr(module, "render"):
            print(f"[ERROR] Missing render() in {module.__name__}")
            return 1
    print("[OK] smoke_test_frontend_import passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
