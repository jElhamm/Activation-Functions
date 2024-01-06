#******************************************************************************************************
#    Class representing the PReLU (Parametric rectified linear unit) activation function              *
#                                                                                                     *
#    The formula implemented in the code for the PReLU activation function is:                        *
#                                                                                                     *
#  ---> For the activation function:             f(x) = { alpha * x, if x < 0                         *
#                                                         x,         if x >= 0 },                     *
#                                                                                                     *
#       where alpha is the parameter that determines the slope for negative values.                   *
#                                                                                                     *
#  ---> For the derivative of the activation function:       f'(x) = { alpha, if x < 0                *
#                                                                       1,    if x >= 0 },            *
#                                                                                                     *
#       which is a piecewise function that depends on the value of x.                                 *
#                                                                                                     *
# ---> This implementation allows the PReLU activation function to have different slopes              *
#      for negative values (alpha * x) compared to positive values (x), providing more                *
#      flexibility in modeling the activation behavior.                                               *
#                                                                                                     *
#******************************************************************************************************



import numpy as np

class PReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def activation(self, x):
        """
        Compute the PReLU activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the PReLU activation function.
        """
        return np.where(x < 0, self.alpha * x, x)

    def derivative(self, x):
        """
        Compute the derivative of the PReLU activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Derivative of the PReLU activation function.
        """
        return np.where(x < 0, self.alpha, 1)