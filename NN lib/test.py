# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 12:19:35 2022

@author: rfigg
"""

import activations as act
import numpy as np

ar = np.array([[-1,1.1042286,0.1022536,1.46509147]])
print(ar)
print(act.ReLU().activation(ar))