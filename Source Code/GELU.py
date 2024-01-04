#*******************************************************************************************************************************
#       Class representing the GELU (Gaussian Error Linear Unit) activation function                                           *
#                                                                                                                              *
# ---> The GELU (Gaussian Error Linear Unit) activation function is defined as:                                                *
#                                                                                                                              *
#       𝐺𝐸𝐿𝑈(𝑥) = 0.5 * 𝑥 * (1 + tanh(√(2/𝜋) * (𝑥 + 0.044715 * 𝑥^3)))                                                         *
#                                                                                                                              *
# ---> The derivative of the GELU activation function, using the derivative approximation used by PyTorch, is defined as:      *
#                                                                                                                              *
#        𝐷𝐺𝐸𝐿𝑈(𝑥) = 0.5 * (1 + 𝑐𝑑𝑓 + 𝑥 * (1 - 𝑐𝑑𝑓))                                                                           *
#                                                                                                                              *
# ---> Where 𝑐𝑑𝑓 represents the cumulative distribution function (CDF) and is computed as:                                     *
#                                                                                                                              *
#       𝑐𝑑𝑓 = 0.5 * (1 + tanh(√(2/𝜋) * (𝑥 + 0.044715 * 𝑥^3) + (𝑥 / √2)))                                                       *
#                                                                                                                              *
# ---> Note that the formulas use mathematical notation to represent the functions,                                            *
#      such as 𝑥^3 for x raised to the power of 3 and 𝜋 for pi.                                                                *
# #*****************************************************************************************************************************



import numpy as np

class GELU:
    def activation(self, x):
        """
        Compute the Gaussian Error Linear Unit (GELU) activation function for the input x.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output of the GELU activation function.
        """

        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))

    def derivative(self, x):
        """
        Compute the derivative of the GELU activation function for the input x.
        Note: Derivative approximation used by PyTorch

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Derivative of the GELU activation function.
        """

        cdf = 0.5 * (1 + np.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))) + (x / np.sqrt(2))))
        return 0.5 * (1 + cdf + x * (1 - cdf))