"""Exceptions."""


class InvalidLayerNumberError(Exception):
    """Impossible to access the given layer number."""


class CoordinatesInstanciationError(Exception):
    """Exception raised when instantiating coordinates."""


class UnsetStencilError(Exception):
    """Unset Stencil."""


class UnsetTimestepError(Exception):
    """When alpha is not yet set."""


class InvalidSavingFileError(Exception):
    """Raised when wrong file are given as save files."""


class InvalidLayersDefinitionError(Exception):
    """Raised when given layers are invalid."""


class InvalidModelParameterError(Exception):
    """Raised when trying to pass incorrect Parameter to a model."""


class IncoherentWithMaskError(Exception):
    """Raised when value don't match the mask."""


class UnitError(Exception):
    """Exception for non-matching units."""


class UnsetAError(Exception):
    """When A is not yet set."""


class UnsetSigmaError(Exception):
    """When the property sigma is not yet set."""


class UnsetValuesError(Exception):
    """When the property values is not yet set."""


class UnsetCentersError(Exception):
    """When the property centers is not yet set."""


class InappropriateShapeError(Exception):
    """When a tensor has an inappropriate shape."""


class UnmatchingShapesError(Exception):
    """When a tensor has an inappropriate shape."""


class ParallelSlicingError(Exception):
    """When parallel slicing arguments are invalid."""
