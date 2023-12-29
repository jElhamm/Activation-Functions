#************************************************************************************************
# Class representing the Swish activation function                                              *
#                                                                                               *
# ---> The Swish activation function is defined as:                                             *
#                                   Swish(x) = x * (1 / (1 + exp(-x)))                          *
#                                                                                               *
# ---> And the derivative of the Swish function is calculated as:                               *
#                                   Swish'(x) = Swish(x) + (1 - Swish(x)) / (1 + exp(-x))       *
#************************************************************************************************



import numpy as np

class Swish:
    def activation(self, x):
        return x * (1 / (1 + np.exp(-x)))

    def derivative(self, x):
        # Compute the swish activation function
        swish_x = self.activation(x)
        
        # Compute the derivative of the swish activation function
        derivative_x = swish_x + (1 - swish_x) / (1 + np.exp(-x))
        return derivative_x