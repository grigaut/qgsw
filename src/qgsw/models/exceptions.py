"""Models-related exceptions."""


class InvalidSavingFileError(Exception):
    """Raised when wrong file are given as save files."""


class InvalidLayersDefinitionError(Exception):
    """Raised when given layers are invalid."""


class InvalidModelParameterError(Exception):
    """Raised when trying to pass incorrect Parameter to a model."""


class IncoherentWithMaskError(Exception):
    """Raised when value don't match the mask."""
