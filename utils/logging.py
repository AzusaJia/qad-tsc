from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    log_path: str | Path | None = None,
    mode: str = "w",
) -> logging.Logger:
    """Configure file-only logging so runs stay silent on the console."""
    logger = logging.getLogger("qad")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger
