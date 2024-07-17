"""Collinearity Coefficient Configuration."""

from pathlib import Path

from qgsw.configs import keys
from qgsw.configs.base import _Config
from qgsw.configs.exceptions import UnauthorizedAttributeError


class CollinearityCoefficientConfig(_Config):
    """Collinearity Coefficient Configuration."""

    section: str = keys.COEF["section"]
    _type: str = keys.COEF["type"]
    _value: str = keys.COEF["value"]
    _file: str = keys.COEF["source file"]

    @property
    def type(self) -> str:
        """Coefficient type."""
        return self.params[self._type]

    @property
    def value(self) -> float:
        """Constant coefficient value."""
        if self.type != "constant":
            msg = "This attribute is only relevant for constant coefficients."
            raise UnauthorizedAttributeError(msg)
        return self.params[self._value]

    @property
    def source_file(self) -> Path:
        """Source file for changing coefficients."""
        if self.type != "changing":
            msg = "This attribute is only relevant for changing coefficients."
            raise UnauthorizedAttributeError(msg)
        return Path(self.params[self._file])
