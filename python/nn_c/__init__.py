"""
nn_c: Neural network library with C/CUDA backend
"""

# Re-export the C extension module for clean imports
from ._nn_core import Tensor

__all__ = ["Tensor"]
__version__ = "0.1.0"
