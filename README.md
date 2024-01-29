# Activation Functions

   This repository contains implementations of various [*Activation Functions*](https://en.wikipedia.org/wiki/Activation_function) commonly used in neural networks.
   These activation functions provide non-linear transformations to introduce non-linearity and enable the 
   network to learn complex patterns and relationships.


   The following activation functions are included:

   * [ELU  : Exponential Linear Unit](Source%20Code/ELU.py)
   * [Tanh : Hyperbolic Tangent](Source%20Code/Tanh.py)
   * [GELU : Gaussian Error Linear Unit](Source%20Code/GELU.py)
   * [ReLU : Rectified Linear Unit](Source%20Code/ReLU.py)
   * [SELU : Scaled Exponential Linear Unit](Source%20Code/SELU.py)
   * [SMHT : Smooth Hard-Tanh](Source%20Code/SMHT.py)
   * [PReLU : Parametric Rectified Linear Unit](Source%20Code/PReLU.py)
   * [LeakyReLU : Leaky Rectified Linear Unit](Source%20Code/LeakyReLU.py)
   * [Swish activation function](Source%20Code/Swish.py)
   * [Maxout activation function](Source%20Code/Maxout.py)
   * [Sigmoid activation function](Source%20Code/Sigmoid.py)
   * [Gaussian activation function](Source%20Code/Gaussian.py)


## Introduction

   This repository provides efficient and easy-to-use implementations of popular activation functions for deep learning models. 
   Activation functions play a crucial role in introducing non-linearity to neural networks and enhancing their expressive power.

   This repository is organized into individual files, each of which contains the implementation of a specific activation function. 
   The activation functions are implemented in a modular and reusable manner, allowing you to easily incorporate them into your own deep learning projects.

## Usage

   Each activation function is implemented in its own module and can be used independently. 
   Here's a brief overview of each function:

   - **ELU**: The Exponential Linear Unit (ELU) activation function introduces non-linearity with a continuous gradient for negative inputs. It helps mitigate the issue of "dying neurons" encountered in ReLU.


   - **GELU**: The Gaussian Error Linear Unit (GELU) activation function approximates the Gaussian cumulative distribution function. It has been shown to perform well in certain deep learning models, such as Transformers.


   - **Gaussian**: The Gaussian activation function applies an element-wise Gaussian distribution to the input data, acting as a smooth transition between positive and negative values.


   - **LeakyReLU**: The Leaky Rectified Linear Unit (LeakyReLU) activation function allows a small, non-zero gradient for negative inputs. It helps address the dead neuron problem observed in the standard ReLU.


   - **Maxout**: The Maxout activation function takes the maximum activation among a group of linear activations, allowing the network to learn piecewise linear functions.


   - **PReLU**: The Parametric Rectified Linear Unit (PReLU) activation function introduces learnable parameters to control the slope of negative inputs. It helps the network adapt to different types of data.


   - **ReLU**: The Rectified Linear Unit (ReLU) activation function sets negative values to zero while preserving positive values. It is one of the most widely used activation functions due to its simplicity and effectiveness.


   - **SELU**: The Scaled Exponential Linear Unit (SELU) activation function is designed to maintain the mean and variance of the input data during forward propagation, promoting self-normalization and stabilizing the network.


   - **SMHT**: The Smooth Hard-Tanh (SMHT) activation function is a smooth approximation of the hard-tanh function, which is equivalent to clipping the input data between -1 and 1.


   - **Sigmoid**: The Sigmoid activation function maps the input to a value between 0 and 1. It is commonly used in binary classification tasks, providing a smooth transition between classes.


   - **Swish**: The Swish activation function applies a smooth and non-monotonic function that extrapolates the ReLU behavior for positive inputs and saturates for negative inputs.


   - **Tanh**: The Hyperbolic Tangent (Tanh) activation function maps the input to values between -1 and 1. It is commonly used in neural networks to introduce non-linearity.


   * [*More Information*](https://www.geeksforgeeks.org/activation-functions-neural-networks/)


## License

   * [Activation Functions in Neural Networks](https://www.v7labs.com/blog/neural-networks-activation-functions)
   * [The Sigmoid Activation Function - Python Implementation](https://www.digitalocean.com/community/tutorials/sigmoid-activation-function-python)
   * [How to Choose an Activation Function for Deep Learning](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/#:~:text=The%20hyperbolic%20tangent%20activation%20function,the%20range%20%2D1%20to%201.)