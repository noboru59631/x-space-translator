"""Run the Render bridge on the injected PORT."""

from __future__ import annotations

import os

import uvicorn


def configured_port() -> int:
    port = int(os.getenv("PORT", "10000"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    return port


if __name__ == "__main__":
    uvicorn.run(
        "render_bridge_app.main:app",
        host="0.0.0.0",
        port=configured_port(),
    )
