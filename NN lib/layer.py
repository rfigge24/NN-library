# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 19:04:18 2022

@author: rfigg
"""


class Layer:
    def __init__(self):
        self.input = None
        self.output = None
 #   
    def forward_propagation(self, input_data):
        #TODO: return output
        print(input_data)
        pass
 #   
    def backward_propagation(self, output_gradient, learning_rate):
        #TODO: update parameters and return input gradient
        pass


