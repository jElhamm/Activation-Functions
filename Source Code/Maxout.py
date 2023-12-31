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
 
    def derivative(self, x):
        if x.ndim > 1:
            axis = tuple(range(x.ndim - 1))                                                 # Get the axes to apply the max operation on
            axis = axis[0] if isinstance(axis, tuple) else axis                             # Convert axis to a single value if it's a tuple
            max_indices = np.argmax(x, axis=axis)                                           # Get the indices of the maximum values along the specified axis
            d = np.zeros_like(x)                                                            # Create an array of zeros with the same shape as x
            np.put_along_axis(d, np.expand_dims(max_indices, axis=axis), 1, axis=axis)      # Set the value 1 at the indices of the maximum values
            return d                                                                        # Return the derivative array
        else:
            # If x is 1D, return an array with 1s at the maximum value index and 0s elsewhere
            return np.where(x == np.max(x), 1, 0)
 