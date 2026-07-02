# -*- coding: utf-8 -*-
"""Legacy frontend entrypoint.

Use ``frontend/app_gradio.py`` for the course-document startup command. This
file proxies to the same builder for backward compatibility.
"""

from __future__ import annotations

from frontend.app_gradio import build_app


if __name__ == "__main__":
    app = build_app()
    if hasattr(app, "launch"):
        app.launch()
    else:
        print(app)
