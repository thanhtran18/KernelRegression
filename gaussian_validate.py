import utils as utils
import numpy as np
import matplotlib.pyplot as plt
import math as math




def evaluate_gaussian_kernel(X_test, t_test, X_train, t_train, h):
    G = (1 / math.sqrt(2 * math.pi * h**2)) * np.exp(-utils.dist2(X_test.reshape(X_test.shape[0], 1), X_train) / (2 * h**2))
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
# X_test = X_n[np.arange(TRAIN_SIZE, X_n.shape[0]), :]  # testing input data
# t_test = t[np.arange(TRAIN_SIZE, X_n.shape[0])]  # testing output data
# t_test = t_test.reshape(t_test.shape[0], 1)
t_train = t_train.reshape(t_train.shape[0], 1)

k_folds = 10
cv_steps = np.arange(0, TRAIN_SIZE, math.ceil(TRAIN_SIZE / k_folds))
if cv_steps[cv_steps.shape[0] - 1] != TRAIN_SIZE:
    cv_steps = np.append(cv_steps, TRAIN_SIZE + 1)
    # cv_steps[cv_steps.shape[0]] = TRAIN_SIZE + 1

# TODO: REMOVE 0.001
# Bandwidths to explore in cross-validation.
hs = [0.001, 0.01, 0.1, 0.25, 1, 2, 3, 4]
# valid_err stores average root mean squared error over all folds, for each lambda.
valid_err = []
for h_i in np.arange(len(hs)):
    # Cross-validation
    # v_err accumulates validation error for this lambda setting over all CV folds.
    v_err = 0
    for c_i in np.arange(k_folds):
        test_inds = np.arange(cv_steps[c_i], (cv_steps[c_i + 1]))
        train_inds = np.setdiff1d(np.arange(TRAIN_SIZE), test_inds)
        # train_inds = set(np.arange(TRAIN_SIZE)).difference(set(test_inds))
        X_trainu = X_train[train_inds - 1, :]
        X_testu = X_train[test_inds - 1, :]
        t_trainu = t_train[train_inds - 1, :]
        t_testu = t_train[test_inds - 1, :]
        [t_est, tt_err] = evaluate_gaussian_kernel(X_testu, t_testu, X_trainu, t_trainu, hs[h_i])
        v_err = v_err + tt_err
    valid_err.append(v_err)

# Produce plot of results.
# plt.figure(101)
# set(gca, 'FontSize', 15)
# Plot regression results.
print(valid_err)
plt.semilogx(hs, valid_err, 'bo-')
plt.xlabel('h value')
plt.ylabel('Validation set error')
plt.title('Cross-validation for bandwidth with Gaussian kernel regression')
plt.show()

# plt.semilogx(xLabel, error)
# plt.ylabel('Error')
# plt.legend(['Average Validation error'])
# plt.title('Polynomial degree = 8, 10-fold cross validation regularization')
# plt.xlabel('lambda on log scale')
# plt.show()