#*****************************************************************************
#    Class representing the SELU activation function                         *
#                                                                            *
#    ---> The SELU activation function:                                      *
#                                     f(x) = scale * (x, if x >= 0,          *
#                                     alpha * (e^x - 1), if x < 0)           *
#                                                                            *  
#    ---> The derivative of the SELU function:                               *
#                                     f'(x) = scale * (1, if x >= 0,         *
#                                     alpha * e^x, if x < 0)                 *
#                                                                            *
#*****************************************************************************


import numpy as np


class SELU:
    def __init__(self, alpha=1.6732632423543772848170429916717, scale=1.0507009873554804934193349852946):
        self.alpha = alpha
        self.scale = scale

    def activation(self, x):
        # Apply SELU activation function to input x
        return self.scale * np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

    def derivative(self, x):
        # Calculate the derivative of SELU activation function with respect to x
        return self.scale * np.where(x >= 0, 1, self.alpha * np.exp(x))