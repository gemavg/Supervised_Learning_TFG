# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:59:41 2022

@author: usuario
"""

import numpy as np
import seaborn as sns
from sklearn.svm import SVR
import matplotlib.pyplot as plt 
import pandas as pd 



np.random.seed(1)
#generar datos
X = np.sort(5 * np.random.rand(400, 1), axis=0)
y = np.sin(X).ravel()


# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(80))


mse = ([0.18877734378428598, 0.19274779313537632, 0.23103610992748752], [0.17733419787099913, 0.17957605760826234, 0.19027830590421962], [0.17601047087357302, 0.17530981495988146, 0.19495488911875575])
ax = sns.heatmap(mse, xticklabels=False, yticklabels=False)
ax.set_xlabel('0.05                         0.1                         0.3') 
ax.set_ylabel('0.9                         0.1                         0.01')

##ax.set_xlabel('epsilon') 
##ax.set_ylabel('gamma')

