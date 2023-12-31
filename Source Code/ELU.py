#**********************************************************************************************************
#    Class representing the ELU activation function                                                       *
#                                                                                                         *
#   ---> The ELU activation function:                                                                     *
#                                   f(x) = x if x >= 0                                                    *
#                                   alpha * (exp(x) - 1) if x < 0                                         *
#                                                                                                         *
#   ---> The derivative of the ELU activation function:                                                   *
#                                   f'(x) = 1 if x >= 0                                                   *
#                                   f(x) + alpha if x < 0                                                 *
#                                                                                                         *
#   ---> In these formulas, "x" represents the input to the activation function,                          *
#        and "alpha" is a parameter that controls the behavior of the function for negative inputs.       *
#                                                                                                         *
#**********************************************************************************************************


import numpy as np

class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def activation(self, x):
        # Apply ELU activation function element-wise
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x):
        # Compute the derivative of the ELU activation function
        return np.where(x >= 0, 1, self.activation(x) + self.alpha)