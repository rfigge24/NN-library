# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 20:00:07 2022

@author: rfigg
"""
import numpy as np
from loss import Loss_Function

#Mean Squared Error Loss:
MSE = Loss_Function(
        lambda y_true, y_pred: np.mean(np.power(y_true - y_pred,2)),  
        lambda y_true, y_pred: 2 * (y_pred - y_true) / np.size(y_true)
        )
