"""
Logging system for KADAS AI Segmentation Plugin.

Provides structured logging with multiple levels and automatic stacktrace
collection for critical errors. Log file location is configurable via
environment variable.
"""

import logging
import os
import sys
import traceback

LOG_LEVELS = {
    "STANDARD": logging.INFO,
    "DEBUG": logging.DEBUG,
    "ERRORS": logging.ERROR,
    "WARNING": logging.WARNING,
    "CRITICAL": logging.CRITICAL,
}


class CriticalFileHandler(logging.FileHandler):
    """
    FileHandler that writes full stacktrace for CRITICAL errors.
    """

    def emit(self, record):
        if record.levelno >= logging.CRITICAL and record.exc_info:
            record.msg = f"{record.msg}\n{''.join(traceback.format_exception(*record.exc_info))}"
        super().emit(record)


def get_logger(level="DEBUG", log_to_console=False):
    """
    Returns a configured logger for the plugin.
    
    Args:
        level: Log level ('STANDARD', 'DEBUG', 'ERRORS', 'WARNING', 'CRITICAL')
        log_to_console: Also log to console (useful for debugging)
    
    Returns:
        Configured logger instance
    
    Log file location:
        - Environment variable: KADAS_AI_SEGMENTATION_LOG
        - Default: ~/.kadas/ai_segmentation.log
    """
    log_path = os.environ.get('KADAS_AI_SEGMENTATION_LOG', os.path.expanduser('~/.kadas/ai_segmentation.log'))
    log_dir = os.path.dirname(log_path)
    try:
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass

    logger = logging.getLogger('kadas_ai_segmentation')
    logger.propagate = False  # Avoid double logging if root logger is configured

    # Remove duplicate handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # File handler with stacktrace for CRITICAL
    try:
        fh = CriticalFileHandler(log_path, mode='a', encoding='utf-8')
        fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass

    # Optional console handler
    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    logger.debug(f"Logger initialized with level {level.upper()} (file: {log_path})")
    return logger
