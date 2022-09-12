# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:17:58 2022

@author: usuario
"""

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


np.random.seed(1)
#generar datos
X = np.sort(5 * np.random.rand(400, 1), axis=0)
y = np.sin(X).ravel()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(80))



#ajustar modelo de regresion
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)


#mirar resultado
lw = 2

svrs = [svr_rbf, svr_lin, svr_poly]
kernel_label = ["RBF", "Linear", "Polynomial"]
model_color = ["r", "b", "g"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(
        X,
        svr.fit(X, y).predict(X),
        color=model_color[ix],
        lw=lw,
        label="{} model".format(kernel_label[ix]),
    )
    axes[ix].scatter(
        X[svr.support_],
        y[svr.support_],
        facecolor=model_color[ix],
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



y_predrbf = svr_rbf.predict(X)
mse_rbf = (np.square(y - y_predrbf)).mean()
print(mse_rbf)

y_predlin = svr_lin.predict(X)
mse_lin = (np.square(y - y_predlin)).mean()
print(mse_lin)

y_predpoly = svr_poly.predict(X)
mse_poly = (np.square(y - y_predpoly)).mean()
print(mse_poly)


