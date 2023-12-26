#************************************************************************************************************
# Class representing the sigmoid activation function                                                        *
# The sigmoid function is defined as:                                   f(x)  = 1 / (1 + exp(-x))           *
# The derivative of the sigmoid function can be computed as:            f'(x) = f(x) * (1 - f(x))           *
#************************************************************************************************************


import numpy as np

class Sigmoid:
    def activation(self, x):
        """
        Compute the sigmoid activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        """
        Compute the derivative of the sigmoid activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Derivative of the sigmoid activation function.
        """
        sig_x = self.activation(x)
        return sig_x * (1 - sig_x)