#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 6 22:04:25 2020

@author: alfredocu
"""

import numpy as np
# import numpy.random as rnd
import matplotlib.pyplot as plt

np.random.seed(42)
m = 300
r = 0.5
ruido = r * np.random.randn(m, 1)
x = 6 * np.random.rand(m, 1) - 3
y = 0.5 * x**2 + x + 2 + ruido

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y)

###############################################################################

# plt.plot(xtrain, ytrain, "b.")
# plt.plot(xtest, ytest, "r.")
# plt.xlabel("$x$", fontsize = 18)
# plt.ylabel("$y$", fontsize = 18)
# plt.axis([-3, 3, 0, 10])
# plt.show()
# plt.savefig("TT.eps", format="eps")

###############################################################################

# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor(max_depth = 4) # 2 1 4 10
# model.fit(xtrain, ytrain)

# print("Train: ", model.score(xtrain, ytrain))
# print("Test: ", model.score(xtest, ytest))

# x_new = np.linspace(-3, 3, 50).reshape(-1, 1)
# y_pred = model.predict(x_new)

# plt.plot(x_new, y_pred, "k-", linewidth= 3)
# plt.plot(xtrain, ytrain, "b.")
# plt.plot(xtest, ytest, "r.")
# plt.xlabel("$x$", fontsize = 18)
# plt.ylabel("$y$", fontsize = 18)
# plt.axis([-3, 3, 0, 10])
# plt.show()
# plt.savefig("Tree.eps", format="eps")

###############################################################################

# from sklearn.neighbors import KNeighborsRegressor
# model = KNeighborsRegressor(n_neighbors = 15, weights = "uniform") # 2 100
# model.fit(xtrain, ytrain)

# print("Train: ", model.score(xtrain, ytrain))
# print("Test: ", model.score(xtest, ytest))

# x_new = np.linspace(-3, 3, 100).reshape(-1, 1) # 1000
# y_pred = model.predict(x_new)

# plt.plot(x_new, y_pred, "k-", linewidth = 3)
# plt.plot(xtrain, ytrain, "b.")
# plt.plot(xtest, ytest, "r.")
# plt.xlabel("$x$", fontsize = 18)
# plt.ylabel("$y$", fontsize = 18)
# plt.axis([-3, 3, 0, 10])
# plt.show()
# plt.savefig("KN.eps", format="eps")

###############################################################################

# from  sklearn.svm import SVR
# model = SVR(gamma = "scale", C = 10, epsilon = 0.1, kernel = "rbf") # linear rbf
# model.fit(xtrain, ytrain.ravel())

# print("Train: ", model.score(xtrain, ytrain))
# print("Test: ", model.score(xtest, ytest))

# x_new = np.linspace(-3, 3, 1000).reshape(-1, 1) # 1000
# y_pred = model.predict(x_new)

# plt.plot(x_new, y_pred, "k-", linewidth = 3)
# plt.plot(xtrain, ytrain, "b.")
# plt.plot(xtest, ytest, "r.")
# plt.xlabel("$x$", fontsize = 18)
# plt.ylabel("$y$", fontsize = 18)
# plt.axis([-3, 3, 0, 10])
# plt.show()
# plt.savefig("SVR.eps", format="eps")

###############################################################################

# from  sklearn.kernel_ridge import KernelRidge
# model = KernelRidge(alpha = 0.1, kernel = "rbf") # linear rbf
# model.fit(xtrain, ytrain.ravel())

# print("Train: ", model.score(xtrain, ytrain))
# print("Test: ", model.score(xtest, ytest))

# x_new = np.linspace(-3, 3, 1000).reshape(-1, 1) # 1000
# y_pred = model.predict(x_new)

# plt.plot(x_new, y_pred, "k-", linewidth = 3)
# plt.plot(xtrain, ytrain, "b.")
# plt.plot(xtest, ytest, "r.")
# plt.xlabel("$x$", fontsize = 18)
# plt.ylabel("$y$", fontsize = 18)
# plt.axis([-3, 3, 0, 10])
# plt.show()
# plt.savefig("KR.eps", format="eps")

###############################################################################

from  sklearn.neural_network import MLPRegressor
model = MLPRegressor(hidden_layer_sizes = (100,20), solver = "adam", activation = "relu", batch_size = 10) # linear rbf
model.fit(xtrain, ytrain.ravel())

print("Train: ", model.score(xtrain, ytrain))
print("Test: ", model.score(xtest, ytest))

x_new = np.linspace(-3, 3, 1000).reshape(-1, 1) # 1000
y_pred = model.predict(x_new)

plt.plot(x_new, y_pred, "k-", linewidth = 3)
plt.plot(xtrain, ytrain, "b.")
plt.plot(xtest, ytest, "r.")
plt.xlabel("$x$", fontsize = 18)
plt.ylabel("$y$", fontsize = 18)
plt.axis([-3, 3, 0, 10])
# plt.show()
plt.savefig("MLPR.eps", format="eps")