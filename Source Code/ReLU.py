#************************************************************************************************
# Class representing the ReLU activation function                                               *
#                                                                                               *
# ---> The ReLU (Rectified Linear Unit) activation function is defined as follows:              *
#       - For any input x, if x is greater than 0, the output is equal to x.                    *
#       - Otherwise, if x is less than or equal to 0, the output is 0.                          *
#                                                                                               *
# ---> Mathematical notation:                                                                   *
#       - Let f(x) be the ReLU activation function.                                             *
#       - f(x) = max(0, x)                                                                      *
#************************************************************************************************


import numpy as np


class ReLU:
    def activation(self, x):
        # Returns the element-wise maximum of x and 0.
        return np.maximum(0, x)

    def derivative(self, x):
        # Returns 1 where the input x is greater than 0, and 0 otherwise.
        return np.where(x > 0, 1, 0)