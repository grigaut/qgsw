"""Coefficient instantiation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union, overload

from qgsw.fields.variables.coefficients.coef_names import CoefficientName
from qgsw.fields.variables.coefficients.core import (
    NonUniformCoefficient,
    SmoothNonUniformCoefficient,
    UniformCoefficient,
)

if TYPE_CHECKING:
    from qgsw.configs.models import (
        CoefConfig,
        ModelConfig,
        NonUniformCoefConfig,
        SmoothNonUniformCoefConfig,
        UniformCoefConfig,
    )
    from qgsw.configs.space import SpaceConfig

CoefType = Union[
    UniformCoefficient,
    NonUniformCoefficient,
    SmoothNonUniformCoefficient,
]


@overload
def instantiate_coef(
    model_config: ModelConfig[UniformCoefConfig],
    space_config: SpaceConfig,
) -> UniformCoefficient: ...


@overload
def instantiate_coef(
    model_config: ModelConfig[NonUniformCoefConfig],
    space_config: SpaceConfig,
) -> NonUniformCoefficient: ...


@overload
def instantiate_coef(
    model_config: ModelConfig[SmoothNonUniformCoefConfig],
    space_config: SpaceConfig,
) -> SmoothNonUniformCoefficient: ...


@overload
def instantiate_coef(
    model_config: ModelConfig[CoefConfig],
    space_config: SpaceConfig,
) -> CoefType: ...
@overload
def instantiate_coef(
    model_config: ModelConfig,
    space_config: SpaceConfig,
) -> CoefType: ...


def instantiate_coef(
    model_config: ModelConfig
    | ModelConfig[CoefConfig]
    | ModelConfig[UniformCoefConfig]
    | ModelConfig[NonUniformCoefConfig]
    | ModelConfig[SmoothNonUniformCoefConfig],
    space_config: SpaceConfig,
) -> CoefType:
    """Instantiate the coefficient.

    Args:
        model_config (ModelConfig[CoefConfig]): Model Configuration.
        space_config (SpaceConfig): Space Configuration.

    Raises:
        ValueError: If the coefficient is not valid.

    Returns:
        CoefType: Coefficient
    """
    coef_config = model_config.collinearity_coef
    if coef_config.type == CoefficientName.UNIFORM:
        coef = UniformCoefficient.from_config(
            space_config=space_config,
        )
        coef.update(coef_config.initial)
    elif coef_config.type == CoefficientName.NON_UNIFORM:
        coef = NonUniformCoefficient.from_config(
            space_config=space_config,
        )
        coef.update(coef_config.matrix)
    elif coef_config.type == CoefficientName.SMOOOTH_NON_UNIFORM:
        coef = SmoothNonUniformCoefficient.from_config(
            space_config=space_config,
        )
        coef.sigma = coef_config.sigma
        coef.update(coef_config.initial, coef_config.centers)
    else:
        msg = "Possible coefficient types are: "
        coef_types = [
            UniformCoefficient.get_name(),
            NonUniformCoefficient.get_name(),
            SmoothNonUniformCoefficient.get_name(),
        ]
        msg += ", ".join(coef_types)
        raise ValueError(msg)
    return coef
