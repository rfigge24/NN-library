# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:40:24 2022

@author: rfigg
"""

from layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
 #
    def forward_propagation(self, input_data):
        self.input = input_data
        return self.activation(self.input)
 #
    def backward_propagation(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
        
