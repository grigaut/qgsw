"""Models-related exceptions."""


class InvalidSavingFileError(Exception):
    """Raised when wrong file are given as save files."""


class InvalidLayersDefinitionError(Exception):
    """Raised when given layers are invalid."""
