"""Centralised Loguru logger for the entire pipeline."""
import sys
from pathlib import Path

from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str = "titanic-mlops"):
    """Return a configured logger instance."""
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{name}</cyan> — {message}")
    logger.add(
        LOG_DIR / "pipeline.log",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} — {message}",
    )
    return logger.bind(name=name)
