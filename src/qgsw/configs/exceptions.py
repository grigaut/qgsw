"""Configuration Related Exceptions."""


class ConfigError(Exception):
    """Configuration-Related Error."""


class UnexpectedFieldError(Exception):
    """Raised when trying to access an unexpected configuration field."""
