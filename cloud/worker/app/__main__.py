"""Run the worker with the port injected by Cloud Run."""

from __future__ import annotations

import os

import uvicorn


def configured_port() -> int:
    port = int(os.getenv("PORT", "8080"))
    if not 1 <= port <= 65535:
        raise ValueError("PORT must be between 1 and 65535")
    return port


if __name__ == "__main__":
    uvicorn.run(
        "cloud.worker.app.main:app",
        host="0.0.0.0",
        port=configured_port(),
    )
