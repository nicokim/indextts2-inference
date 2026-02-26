import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    level = os.environ.get("INDEXTTS_LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, level, logging.WARNING))
    return logger
