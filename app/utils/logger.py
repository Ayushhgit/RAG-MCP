import logging
import sys
from pathlib import Path

def setup_logger(name: str = "rag-mcp", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger if not already added
    if not logger.handlers:
        logger.addHandler(console_handler)

    return logger

# Global logger instance
logger = setup_logger()