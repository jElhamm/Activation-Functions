#**************************************************************
#    Class representing the ELU activation function           *
#**************************************************************


import numpy as np


class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def activation(self, x):
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x):
        return np.where(x >= 0, 1, self.activation(x) + self.alpha)