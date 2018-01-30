# This sample project will cover what is considered to be a classification neural network.

# Dependencies
import numpy as np  # used for matrix multiplications
import time  # to keep track of application training time

# Variables
n_hidden = 20  # input
n_in = 20  # input
n_out = 20  # output
n_sample = 600  # samples

# Hyper-parameters
learning_rate = 0.01
momentum = 0.9

# Non-deterministic seeding
np.random.seed(0)


# sigmoid turns numbers into probabilities
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2


# Training function
# x = input data
# t = transpose
# v & w = 2 layers of the network
# bv & bw = bias values for each layer
def train(x, t, v, w, bv, bw):
    # forward propagation ... matrix multiply + bias
    A = np.dot(x, v) + bv
    Z = np.tanh(A)

    B = np.dot(Z, w) + bw
    Y = sigmoid(B)

    # backward propagation
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(w, Ew)

    #predicting loss
    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    # can be shortened using tensorflow
    loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 -Y)) # cross-entropy
    