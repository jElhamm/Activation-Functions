#*********************************************************
# Class representing the Tanh activation function        *
#*********************************************************


import numpy as np

class Tanh:
    def activation(self, x):
        return np.tanh(x)
    
        # Implementation without using ready function:
        # return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def derivative(self, x):
        return 1 - np.power(self.activation(x), 2)
    
        # Implementation without using ready function:
        # tanh_x = self.activation(x)
        # return 1 - np.power(tanh_x, 2)