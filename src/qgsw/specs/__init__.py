"""System specs."""

from qgsw.specs._utils import Device

DEVICE = Device.set_automatically()


def use_cpu() -> None:
    """Use CPU."""
    DEVICE.use_cpu()
