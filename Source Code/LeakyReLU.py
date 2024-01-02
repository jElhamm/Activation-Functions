#**********************************************************************
#    Class representing the LeakyReLU activation function             *
#                                                                     *
#   The LeakyReLU activation function:                                *
#                                     f(x) = max(alpha * x, x)        *
#                                                                     *
#   The derivative of the LeakyReLU function:                         *
#                                     f'(x) = 1, if x > 0             *
#                                     f'(x) = alpha, if x <= 0        *
#                                                                     *
#**********************************************************************


import numpy as np

class LeakyReLU:
    def __init__(self, alpha=0.01):
        # Initialize LeakyReLU with a given alpha (slope for negative inputs)
        self.alpha = alpha

    def activation(self, x):
        # Apply LeakyReLU activation function to the input x
        # Replace negative values with alpha * x
        return np.maximum(self.alpha * x, x)

    def derivative(self, x):
        """
          Calculate the derivative of LeakyReLU function at input x
                  ---> If x > 0, derivative is 1
                  ---> If x <= 0, derivative is alpha
        """
        return np.where(x > 0, 1, self.alpha)