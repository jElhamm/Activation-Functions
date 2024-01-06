# This program demonstrates various activation functions used in neural networks.
# It imports different activation functions from separate modules and applies them to user-input values.
# It then prints the activation results and their derivatives.


import numpy as np
from sigmoid import Sigmoid
from tanh import Tanh
from relu import ReLU
from leaky_relu import LeakyReLU
from elu import ELU
from selu import SELU
from maxout import Maxout
from swish import Swish


# Sigmoid
sigmoid = Sigmoid()
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the Sigmoid activation: "))
print(sigmoid.activation(x))
print(sigmoid.derivative(x))

# Tanh
tanh = Tanh()
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the Tanh activation: "))
print(tanh.activation(x))
print(tanh.derivative(x))

# ReLU
relu = ReLU()
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the ReLU activation: "))
print(relu.activation(x))
print(relu.derivative(x))

# LeakyReLU
leaky_relu = LeakyReLU(alpha=0.1)
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the LeakyReLU activation: "))
print(leaky_relu.activation(x))
print(leaky_relu.derivative(x))

# # ELU
elu = ELU(alpha=0.5)
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the ELU activation: "))
print(elu.activation(x))
print(elu.derivative(x))

# SELU
selu = SELU(alpha=1.5, scale=2.0)
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the SELU activation: "))
print(selu.activation(x))
print(selu.derivative(x))

# Maxout
maxout = Maxout(num_pieces=2)
print("------------------------------------------------------------------------------")
x = np.array(input("---> Enter values for the Maxout activation (comma-separated): ").split(','), dtype=float)
print(maxout.activation(x))
print(maxout.derivative(x))

# Swish
swish = Swish()
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the Swish activation: "))
print(swish.activation(x))
print(swish.derivative(x))

# GELU
gelu = GELU()
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the GELU activation: "))
print(gelu.activation(x))
print(gelu.derivative(x))

# SMHT
smht = SMHT()
x = float(input("---> Enter a value for the smht activation: "))
print(smht.activation(x))
print(smht.derivative(x))

# PReLU
prelu = PReLU(alpha=0.01)
print("------------------------------------------------------------------------------")
x = float(input("---> Enter a value for the PReLU activation: "))
print(prelu.activation(x))
print(prelu.derivative(x))

print("------------------------------------------------------------------------------")