# -*- coding: utf-8 -*-
"""
Updated on 1-12-2022

@author: rfigg
"""
from activation import Activation
from layer import Layer
import numpy as np

#Activation functions below can be used only after training:
#Usefull for hard classification after training

class BinaryStep(Activation):
    def __init__(self, threshold):
        #
        def binarystep(x):
            if x >= threshold:
                return 1
            else:
                return 0
        #
        super().__init__(np.vectorize(binarystep,otypes=[float]), None)




#Activation functions below can be used for backpropagating:

class ELU(Activation):
    def __init__(self, alpha):
        #
        if alpha <=0:
            raise ValueError('alpha needs to be greater then 0!')
        #
        def elu(x):
            if x > 0:
                return x
            else:
                return alpha * (np.exp(x) - 1)
        #
        def elu_prime(x):
            if x > 0:
                return 1
            else:
                return alpha * np.exp(x)
        #
        super().__init__(np.vectorize(elu,otypes=[float]), np.vectorize(elu_prime,otypes=[float]))
    
      
class LReLU(Activation):
    def __init__(self, alpha):
        #
        if alpha <=0:
            raise ValueError('alpha needs to be greater then 0!')
        #
        def lrelu(x):
            if x > 0:
                return x
            else:
                return alpha * x
        #
        def lrelu_prime(x):
            if x > 0:
                return 1
            else:
                return alpha
        #
        super().__init__(np.vectorize(lrelu,otypes=[float]), np.vectorize(lrelu_prime,otypes=[float]))


class ReLU(Activation):
    def __init__(self):
        #
        def relu(x):
            return max(0,x)
        #
        def relu_prime(x):
            if x > 0:
                return 1
            else:
                return 0
        #   
        super().__init__(np.vectorize(relu,otypes=[float]), np.vectorize(relu_prime,otypes=[float]))
        
    
class Sigmoid(Activation):
    def __init__(self):
        #
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        #
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        #
        super().__init__(sigmoid,sigmoid_prime)
        

class SoftMax(Layer): #can be optimized using kron product and reshaping
    #
    def forward_propagation(self, input_data):
        tmp = np.exp(input_data)
        self.output = tmp / np.sum(tmp,1,keepdims=True)
        return self.output
    #
    def backward_propagation(self, output_gradient, learning_rate):  
        a = np.eye(self.output.shape[-1])
        temp1 = np.einsum('ij,jk->ijk',self.output,a)
        temp2 = np.einsum('ij,ik->ijk',self.output,self.output)
        jacobien = temp1 - temp2
        
        input_gradient = output_gradient.reshape((output_gradient.shape[0],1,-1)) @ jacobien
        return input_gradient.reshape((input_gradient.shape[0],-1))

class SoftPlus(Activation):
    def __init__(self):
        #
        def softplus(x):
            return np.log(1 + np.exp(x))
        #
        def softplus_prime(x):
            return 1 / (1 + np.exp(-x))
        #
        super().__init__(softplus,softplus_prime)
        
class Tanh(Activation):
    def __init__(self):
        #
        def tanh(x):
            return np.tanh(x)
        #
        def tanh_prime(x): 
            return 1 - np.tanh(x) ** 2
        #
        super().__init__(tanh, tanh_prime)









        