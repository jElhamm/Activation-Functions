#******************************************************************************************************
#    Class representing the Maxout activation function                                                *
#                                                                                                     *
#   ---> Activation function:                                                                         *
#                           maxout(x) = max(x₁, x₂, ..., xₙ)                                           *
#                                                                                                     *
#   --->Derivative of the activation function:                                                        *
#                                                                                                     *
#       - If x is a 1D array:                                                                         *
#                           maxout'(x) = 1, if x is the maximum value in the array 0, otherwise       *
#                                                                                                     *
#       - If x is a multi-dimensional array:                                                          *
#                           maxout'(x) = 1, at the indices of the maximum values along the            *
#                           specified axis 0, elsewhere                                               *
#                                                                                                     *
#******************************************************************************************************



import numpy as np

class Maxout:
    def __init__(self, num_pieces):
        self.num_pieces = num_pieces

    def activation(self, x):
        if x.ndim > 1:
            axis = tuple(range(x.ndim - 1))                                                 # Get the axes to apply the max operation on
        else:
            axis = None
        return np.max(x, axis=axis)                                                         # Apply the max operation on the specified axes
 