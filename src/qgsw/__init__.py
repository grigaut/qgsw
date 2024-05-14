"""QGSW."""

import torch
from dotenv import load_dotenv

# Load Environment variables
load_dotenv()
# Set the seed for reproducibility
torch.random.manual_seed(0)
