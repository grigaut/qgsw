"""System specs."""

import os

DEVICE = os.environ["DEVICE"]  # "cuda" if torch.cuda.is_available() else "cpu"
