"""Logger class."""

import logging
from typing import ParamSpec

P = ParamSpec("P")

DETAIL_LEVEL = 15


class Logger(logging.Logger):
    """Logger with the .detail method."""

    def detail(self, msg: object, *args: P.args, **kwargs: P.kwargs) -> None:
        """Implement the .detail method.

        Args:
            msg (object): Message.
            *args (P.args): Arguments.
            **kwargs (P.kwargs): Keyword arguments.
        """
        if self.isEnabledFor(DETAIL_LEVEL):
            self._log(DETAIL_LEVEL, msg, args, **kwargs)
