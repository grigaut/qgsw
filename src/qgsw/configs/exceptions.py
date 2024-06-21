"""Configuration Related Exceptions."""


class ConfigError(Exception):
    """Configuration-Related Error."""


class ConfigSaveError(Exception):
    """ConfigurationSsave-Related Error."""


class UnexpectedFieldError(Exception):
    """Raised when trying to access an unexpected configuration field."""


class UnauthorizedAttributeError(Exception):
    """raised when trying to access unauthorized attribute."""
