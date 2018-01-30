# This sample project will cover what is considered to be a classification neural network.
# It will predict xor value of a given input

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
    ew = Y - t
    ev = tanh_prime(A) * np.dot(w, ew)

    # predicting loss
    dw = np.outer(Z, ew)
    dv = np.outer(x, ev)

    # can be shortened using tensorflow
    loss = -np.mean(t * np.log(Y) + (1 - t) * np.log(1 - Y))  # cross-entropy

    return loss, (dv, dw, ev, ew)


# END of training
# START of prediction

def predict(x, v, w, bv, bw):
    A = np.dot(x, v) + bv
    B = np.dot(np.tanh(A), w) + bw
    return (sigmoid(B) > 0.5).astype(int)


# creating 2 layers of NN
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

# using bias
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

# input parameters
parameters = [V, W, bv, bw]

# generate data
X = np.random.binomial(1, 0.5, (n_sample, n_in))
T = X ^ 1

# Training
for epoch in range(100):
    err = []
    update = [0] * len(parameters)

    # time the training
    t0 = time.clock()
    # update weights for each data point
    for i in range(X.shape[0]):
        loss, grad = train(X[i], T[i], * parameters)
        # update loss
        for j in range(len(parameters)):
            parameters[j] -= update[j]

        for j in range(len(parameters)):
            update[j] = learning_rate * grad[j] + momentum * update[j]

        err.append(loss)

    print('Epoch: %d, Loss: %8f, Time: %.4fs' %(epoch, np.mean(err), time.clock() - t0))


# NOW PREDICT!
x = np.random.binomial(1, 0.5, n_in)
print('XOR prediction')
print(x)
print(predict(x, *parameters))
