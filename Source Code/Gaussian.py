#******************************************************************************************************
#    Class representing the Gaussian activation function              *
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

class Gaussian:
    def activation(self, x):
        """
        Compute the Gaussian activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the Gaussian activation function.
        """
        return np.exp(-x**2)

    def derivative(self, x):
        """
        Compute the derivative of the Gaussian activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Derivative of the Gaussian activation function.
        """
        return -2 * x * np.exp(-x**2)