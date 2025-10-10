"""Custom formatters."""

import logging
from typing import ClassVar

from qgsw.logging.log_records import LogRecord
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

    def format(self, record: LogRecord) -> str:
        """Format the psecified record as text.

        Args:
            record (logging.LogRecord): Record.

        Returns:
            str: Text.
        """
        timestamp = self.formatTime(record, "%H:%M:%S")
        level_str = self.level_formats.get(record.levelno, "[LOG     ]")
        msg = record.getMessage()
        txt_indent = record.indent
        indent = f"[{timestamp}] {level_str} "
        full_indent = txt_indent + " " * len(indent)
        msg_parts = msg.split("\n")
        return "\n".join(
            [
                indent + txt_indent + msg_parts[0],
                *[full_indent + p for p in msg_parts[1:]],
            ]
        )


class RichFormatter(logging.Formatter):
    """Custom formatter."""

    def format(self, record: LogRecord) -> str:
        """Format the psecified record as text.

        Args:
            record (logging.LogRecord): Record.

        Returns:
            str: Text.
        """
        msg = record.getMessage()
        txt_indent = record.indent
        msg_parts = msg.split("\n")
        return "\n".join([txt_indent + p for p in msg_parts])
