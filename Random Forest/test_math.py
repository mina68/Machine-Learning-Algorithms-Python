# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 22:09:09 2019

@author: Mina
"""

import numpy as np
import numpy.linalg as la

V = np.array([[7,1],
               [-3,0]], dtype=np.float_)

print(la.inv(V))