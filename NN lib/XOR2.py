# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 21:41:40 2022

@author: rfigg
"""

from neural_network import Neural_Network
from dense import Dense
import activations as act
from losses import MSE
import numpy as np

#Dataset
X = np.array([[1,1],[0,0],[0,1],[1,0]])
Y = np.array([[0], [0],[1],[1]]) #(returns 1, returns 0) test for softmax

#
network = [
    Dense(2,4),
    act.ELU(1),
    Dense(4,1),
    act.ELU(1)
    ]
#
np.random.seed(0)
model = Neural_Network(network, MSE)
model.fit(X, Y, 10000, 0.1,shuffle=True)
