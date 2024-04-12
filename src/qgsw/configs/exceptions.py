"""Configuration Related Exceptions."""


class ConfigError(Exception):
    """Configuration-Related Error."""


class UngivenFieldError(Exception):
    """Raised when trying to access ungiven configuration fields."""
