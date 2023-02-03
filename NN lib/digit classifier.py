# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:12:57 2022

@author: rfigg
"""

from neural_network import Neural_Network
import activations as act
from dense import Dense

network = [
    Dense(10,20),
    act.LReLU(),
    Dense(20,20),
    act.LReLU(),
    Dense(20,10),
    act.SoftMax()   
    ]

model = Neural_Network(network)
