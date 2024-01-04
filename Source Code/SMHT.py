#********************************************************************************************************
#       Class representing the SMHT (Soboleva modified hyperbolic tangent) activation function          *
#                                                                                                       *
# ---> SMHT Activation Function:        f(x) = max(0, tanh(x))                                          *
#                                                                                                       *
#                                                                                                       *
# ---> The derivative of the SMHT activation function is a piecewise function defined as follows:       *
#                                       f'(x) = (1 - x^2)     if  -1 <= x <= 1                          *
#                                                0             otherwise                                *
# #******************************************************************************************************



import numpy as np

class SMHT:
    def activation(self, x):
        """
        Compute the Soboleva modified hyperbolic tangent (smht) activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the smht activation function.
        """
        return np.maximum(0, np.tanh(x))

    def derivative(self, x):
        """
        Compute the derivative of the smht activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Derivative of the smht activation function.
        """
        return (np.logical_and(x > 0, x <= 1) + np.logical_and(x <= 0, x >= -1)) * (1 - np.power(x, 2))