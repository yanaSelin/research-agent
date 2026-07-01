"""Centralized logging configuration: file handler only."""

import logging
import os
from enum import Enum
from pathlib import Path

_LOG_DIR = Path(__file__).parent.parent / "logs"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


def setup_logging(name: str = "agent") -> None:
    """Configure root logger with a single file handler.

    No-op if the root logger already has handlers.

    Args:
        name: Log file stem — "agent" → logs/agent.log.
    """
    if logging.root.handlers:
        return

    _LOG_DIR.mkdir(exist_ok=True)
    log_path = _LOG_DIR / f"{name}.log"

    raw_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    try:
        level = getattr(logging, LogLevel(raw_level).value)
    except ValueError:
        level = logging.INFO

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        ],
    )
