"""Centralised logger using loguru."""

from pathlib import Path
import sys

from loguru import logger


def get_logger(log_file: str | None = None):
    """Configure and return a loguru logger instance.

    Parameters
    ----------
    log_file : str | None
        Optional path to write logs to a rotating file.

    Returns
    -------
    loguru.logger
        Configured logger ready to use.
    """
    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> | "
        "<level>{message}</level>",
        level="INFO",
    )

    # File handler (optional)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            rotation="10 MB",
            retention="7 days",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} | {message}",
            level="DEBUG",
        )

    return logger
