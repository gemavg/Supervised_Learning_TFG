# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 16:27:31 2022

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



####################################################################################################

#ajustar modelo de regresion
svr_rbf1 = SVR(kernel="rbf", C=100, gamma=0.01, epsilon=0.05)
svr_rbf2 = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.05)
svr_rbf3 = SVR(kernel="rbf", C=100, gamma=0.9, epsilon=0.05)


#mirar resultado
lw = 2

svrs = [svr_rbf1, svr_rbf2, svr_rbf3]
kernel_label = ["RBF1 model ε=0.05 γ=0.01", "RBF2 model ε=0.05 γ=0.1", "RBF3 model ε=0.05 γ=0.9"]
model_color = ["m", "b", "g"]


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} ".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=50,
        label="other training data",
    )
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="support vectors",
    )
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.04, "x", ha="center", va="center")
fig.text(0.06, 0.5, "y", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()


y_predrbf1 = svr_rbf1.predict(X)
mse_rbf1 = (np.square(y - y_predrbf1)).mean()
print(mse_rbf1)

y_predrbf2 = svr_rbf2.predict(X)
mse_rbf2 = (np.square(y - y_predrbf2)).mean()
print(mse_rbf2)

y_predrbf3 = svr_rbf3.predict(X)
mse_rbf3 = (np.square(y - y_predrbf3)).mean()
print(mse_rbf3)

###################################################################################################







#####################################################################################################

svr_rbf21 = SVR(kernel="rbf", C=100, gamma=0.01, epsilon=0.1)
svr_rbf22 = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_rbf23 = SVR(kernel="rbf", C=100, gamma=0.9, epsilon=0.1)


#mirar resultado
lw = 2

svrs = [svr_rbf21, svr_rbf22, svr_rbf23]
kernel_label = ["RBF1 model ε=0.1 γ=0.01", "RBF2 model ε=0.1 γ=0.1", "RBF3 model ε=0.1 γ=0.9"]
model_color = ["m", "b", "g"]


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} ".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=50,
        label="other training data",
    )
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="support vectors",
    )
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.04, "x", ha="center", va="center")
fig.text(0.06, 0.5, "y", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()




y_predrbf21 = svr_rbf21.predict(X)
mse_rbf21 = (np.square(y - y_predrbf21)).mean()
print(mse_rbf21)

y_predrbf22 = svr_rbf22.predict(X)
mse_rbf22 = (np.square(y - y_predrbf22)).mean()
print(mse_rbf22)

y_predrbf23 = svr_rbf23.predict(X)
mse_rbf23 = (np.square(y - y_predrbf23)).mean()
print(mse_rbf23)
###############################################################################################








###############################################################################################

svr_rbf31 = SVR(kernel="rbf", C=100, gamma=0.01, epsilon=0.3)
svr_rbf32 = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.3)
svr_rbf33 = SVR(kernel="rbf", C=100, gamma=0.9, epsilon=0.3)

#mirar resultado
lw = 2

svrs = [svr_rbf31, svr_rbf32, svr_rbf33]
kernel_label = ["RBF1 model ε=0.3 γ=0.01", "RBF2 model ε=0.3 γ=0.1", "RBF3 model ε=0.3 γ=0.9"]
model_color = ["m", "b", "g"]


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} ".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor="none",
        edgecolor=model_color[ix],
        s=50,
        label="other training data",
    )
    axes[ix].scatter(
        X[np.setdiff1d(np.arange(len(X)), svr.support_)],
        y[np.setdiff1d(np.arange(len(X)), svr.support_)],
        facecolor="none",
        edgecolor="k",
        s=50,
        label="support vectors",
    )
    axes[ix].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

fig.text(0.5, 0.04, "x", ha="center", va="center")
fig.text(0.06, 0.5, "y", ha="center", va="center", rotation="vertical")
fig.suptitle("Support Vector Regression", fontsize=14)
plt.show()



y_predrbf31 = svr_rbf31.predict(X)
mse_rbf31 = (np.square(y - y_predrbf31)).mean()
print(mse_rbf31)

y_predrbf32 = svr_rbf32.predict(X)
mse_rbf32 = (np.square(y - y_predrbf32)).mean()
print(mse_rbf32)

y_predrbf33 = svr_rbf33.predict(X)
mse_rbf33 = (np.square(y - y_predrbf33)).mean()
print(mse_rbf33)

###############################################################################################



#mse =([mse_rbf1,mse_rbf21,mse_rbf31],[mse_rbf2,mse_rbf22,mse_rbf32],[mse_rbf3,mse_rbf23,mse_rbf31])
#print(mse)
#ax = sns.heatmap(mse)






















