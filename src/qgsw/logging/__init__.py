"""Logging-related tools."""

from qgsw.logging._levels import CRITICAL, DEBUG, DETAIL, INFO, WARNING
from qgsw.logging.core import getLogger, setup_root_logger

__all__ = [
    "CRITICAL",
    "DEBUG",
    "DETAIL",
    "INFO",
    "WARNING",
    "getLogger",
    "setup_root_logger",
]
