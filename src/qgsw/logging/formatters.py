"""Custom formatters."""

import logging
from typing import ClassVar

from qgsw.logging.logger import DETAIL_LEVEL


class Formatter(logging.Formatter):
    """Custom formatter."""

    level_formats: ClassVar[dict[int, str]] = {
        logging.CRITICAL: "[CRITICAL]",
        logging.ERROR: "[ERROR   ]",
        logging.WARNING: "[WARNING ]",
        logging.INFO: "[INFO    ]",
        DETAIL_LEVEL: "[DETAIL  ]",
        logging.DEBUG: "[DEBUG   ]",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the psecified record as text.

        Args:
            record (logging.LogRecord): Record.

        Returns:
            str: Text.
        """
        timestamp = self.formatTime(record, "%H:%M:%S")
        level_str = self.level_formats.get(record.levelno, "[LOG     ]")
        msg = record.getMessage()
        return f"[{timestamp}] {level_str} {msg}"
