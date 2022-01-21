"""
function to mearsure how predict is.
we can use it adjust parameters
"""
import numpy as np
from src.tensor import Tensor


class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual)**2)

    def grad(self, predicted: Tensor, actual: Tensor) -> float):
        return 2 * (predicted - actual)
