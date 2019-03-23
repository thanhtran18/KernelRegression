import utils as utils
import numpy as np
import matplotlib.pyplot as plt
import math as math


def evaluate_epanechnikov(X_test, t_test, X_train, t_train, h):
    # G = (1 / math.sqrt(2 * math.pi * h**2)) * np.exp(-utils.dist2(X_test.reshape(X_test.shape[0], 1), X_train) / (2 * h**2))
    G = 0.75 * (1 - utils.dist2(X_test.reshape(X_test.shape[0], 1), X_train) / (h**2))
    G = np.maximum(G, 0)
    T = np.tile(t_train.T, [X_test.shape[0], 1])
    denom = np.sum(G, 1)
    default_value = np.mean(t_train)

    t_est = np.sum(G * T, 1) / (denom + np.spacing(1))
    chosen_indices = np.where(denom < np.spacing(1))
    t_est[chosen_indices] = default_value
    if len(t_test) > 0:
        test_err = np.sqrt(np.mean((t_test - t_est) ** 2))
    else:
        test_err = []
    return t_est, test_err


[t, X] = utils.loadData()
X_n = utils.normalizeData(X)
t = utils.normalizeData(t)

X_n = X_n[:, 2].reshape(X_n.shape[0], 1)

# CREATE THE TRAIN AND TEST SETS:
# ================================
TRAIN_SIZE = 100  # number of training examples
X_train = X_n[np.arange(0, TRAIN_SIZE), :]  # training input data
t_train = t[np.arange(0, TRAIN_SIZE)]  # training output data
X_test = X_n[np.arange(TRAIN_SIZE, X_n.shape[0]), :]  # testing input data
t_test = t[np.arange(TRAIN_SIZE, X_n.shape[0])]  # testing output data
t_test = t_test.reshape(t_test.shape[0], 1)
t_train = t_train.reshape(t_train.shape[0], 1)


h_values = [0.01, 0.1, 0.25, 1, 2, 3, 4]
# hs = [1]

for h in h_values:
    x_ev = np.arange(min(X_train), max(X_train), 0.01)
    [y_ev, test_err] = evaluate_epanechnikov(x_ev, t_test, X_train, t_train, h)

    plt.plot(x_ev, y_ev, 'r.-')
    plt.plot(X_train, t_train, 'gx', markersize=10)
    plt.plot(X_test, t_test, 'bo', markersize=10, mfc='none')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Fit with h=%.3f' % h)
    plt.show()
