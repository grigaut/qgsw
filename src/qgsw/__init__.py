"""QGSW.

Most 3D Tensors are expected to have the following shape: (1, nl, nx, ny).
    - nl: number of layer
    - nx: number of points in the x direction
    - ny: number of points in the y direction
"""

import torch
from dotenv import load_dotenv

from qgsw.logging import setup_root_logger

# Load Environment variables
load_dotenv()
# Set the seed for reproducibility
torch.random.manual_seed(0)
# Logging
setup_root_logger(1)
