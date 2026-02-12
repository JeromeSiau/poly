"""Centralized structlog configuration for all Poly scripts."""

import structlog

_configured = False


def configure_logging() -> None:
    """Configure structlog with the project-standard processor chain.

    Safe to call multiple times; only the first call takes effect.
    """
    global _configured
    if _configured:
        return
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ]
    )
    _configured = True
