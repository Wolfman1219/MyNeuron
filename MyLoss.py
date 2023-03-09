import numpy as np


def MSE_loss(y_true = None, y_pred = None, derivative=False):
    if derivative:
        return 2 * (y_pred - y_true.T) / y_true.T.size
    return np.mean((y_true.T - y_pred) ** 2)


def binary_cross_entropy(y_true = None, y_pred = None, derivative=False):
    epsilon = 1e-10 # small value added to avoid taking log of 0
    if derivative:
        derivative = -(y_true.T / (y_pred + epsilon)) + (1 - y_true.T) / (1 - y_pred + epsilon)
        derivative /= len(y_true.T) # normalize by batch size
        return derivative
    loss = -np.mean(y_true.T * np.log(y_pred + epsilon) + (1 - y_true.T) * np.log(1 - y_pred + epsilon))
    return loss

