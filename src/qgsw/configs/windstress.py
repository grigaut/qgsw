"""Wind Stress configuration."""

from typing import Any

from qgsw.configs import keys
from qgsw.configs.base import _Config, _DataConfig


class WindStressConfig(_Config):
    """WindStress Configuration."""

    section: str = keys.WINDSTRESS["section"]
    _data_section: str = keys.WINDSTRESS_DATA["section"]
    _type: str = keys.WINDSTRESS["type"]
    _magnitude: str = keys.WINDSTRESS["magnitude"]
    _drag_coef: str = keys.WINDSTRESS["drag coefficient"]

    def __init__(self, params: dict[str, Any]) -> None:
        """Instantiate Windstress Configuration.

        Args:
            params (dict[str, Any]): Windstress configuration parameters.
        """
        super().__init__(params)
        self._data = WindStressDataConfig.parse(self.params)

    @property
    def data(self) -> "WindStressDataConfig":
        """Windstress Data Configuration."""
        return self._data

    @property
    def type(self) -> str:
        """Windstress type (cosine, data)."""
        return self.params[self._type]

    @property
    def magnitude(self) -> float:
        """Wind Stress Magnitude."""
        return self.params[self._magnitude]

    @property
    def drag_coefficient(self) -> str:
        """Wind drag coefficient."""
        return self.params[self._drag_coef]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return super()._validate_params(params)


class WindStressDataConfig(_DataConfig):
    """WindStress Data Configuration."""

    section: str = keys.WINDSTRESS_DATA["section"]
    _url: str = keys.WINDSTRESS_DATA["url"]
    _folder: str = keys.WINDSTRESS_DATA["folder"]
    _data_type: str = keys.WINDSTRESS_DATA["data"]
    _lon: str = keys.WINDSTRESS_DATA["longitude"]
    _lat: str = keys.WINDSTRESS_DATA["latitude"]
    _time: str = keys.WINDSTRESS_DATA["time"]
    _1: str = keys.WINDSTRESS_DATA["field 1"]
    _2: str = keys.WINDSTRESS_DATA["field 2"]
    _method: str = keys.WINDSTRESS_DATA["method"]

    @property
    def data_type(self) -> str:
        """Kind of data (speed, tau)."""
        return self.params[self._data_type]

    @property
    def longitude(self) -> str:
        """Longitude field name."""
        return self.params[self._lon]

    @property
    def latitude(self) -> str:
        """Latitude field name."""
        return self.params[self._lat]

    @property
    def time(self) -> str:
        """Time field name."""
        return self.params[self._time]

    @property
    def field_1(self) -> str:
        """Field 1 name."""
        return self.params[self._1]

    @property
    def field_2(self) -> str:
        """Field 2 name."""
        return self.params[self._2]

    @property
    def method(self) -> str:
        """Interpolation method."""
        return self.params[self._method]

    def _validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        return super()._validate_params(params)
