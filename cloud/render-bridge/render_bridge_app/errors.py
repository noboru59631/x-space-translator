"""Safe, classified errors returned by the bridge."""

from __future__ import annotations


class BridgeError(Exception):
    """An expected bridge failure with a stable public code."""

    def __init__(self, code: str, message: str, *, status_code: int = 500) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class MediaValidationError(BridgeError):
    """The remuxed media is not safe to send to Groq."""
