
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:10:14 2022

@author: rfigg
"""

from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(1,output_size)
 #   
    def forward_propagation(self, input_data):
        self.input = input_data
        return np.dot(self.input,self.weights.T) + np.tile(self.bias,(input_data.shape[0],1))
 #
    def backward_propagation(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient.T, self.input)
        input_gradient = np.dot(output_gradient, self.weights)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient,0)
        return input_gradient
        