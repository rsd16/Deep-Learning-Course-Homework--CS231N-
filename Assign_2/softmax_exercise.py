#FILE NAME: softmax_exercise.py########################################################################################
# Softmax exercise
# Complete and hand in the completed code  (including its outputs and any supporting code outside of it)
#
# This exercise is analogous to the SVM exercise. You will:
#   implement a fully-vectorized loss function for the Softmax classifier
#   implement the fully-vectorized expression for its analytic gradient
#   check your implementation with numerical gradient
#   use a validation set to tune the learning rate and regularization strength
#   optimize the loss function with SGD
#   visualize the final learned weights
#
#######################################################################################################################

import random
import numpy as np
import matplotlib.pyplot as plt
from utilities.load_cifar10 import load_CIFAR10
#from load_cifar10 import load_CIFAR10 # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM exercise, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'D:\\safe\\university\\arshad\\deep\\cifar-10-batches-py'

    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    Y_dev = Y_train[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # zero-center the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev


# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
    del X_train, Y_train
    del X_test, Y_test
    print('Clear previously loaded data.')
except:
    pass

# Invoke the above function to get our data.
X_train, Y_train, X_val, Y_val, X_test, Y_test, X_dev, Y_dev = get_CIFAR10_data()

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', Y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', Y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', Y_dev.shape)

########################################################################################################################
#Softmax Classifier
#Your code for this section will all be written inside classifiers/softmax_loss.py.
########################################################################################################################

# First implement the naive softmax loss function with nested loops.
# Open the file classifiers/softmax_loss.py and implement the
# softmax_loss_naive function.

from classifiers.softmax_loss import softmax_loss_naive
#from softmax_loss import softmax_loss_naive # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, Y_dev, 0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print()
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))

########################################################################################################################

# Complete the implementation of softmax_loss_naive and implement a (naive)
# version of the gradient that uses nested loops.
loss, grad = softmax_loss_naive(W, X_dev, Y_dev, 0.0)

# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.

from utilities.gradient_check import grad_check_sparse
#from gradient_check import grad_check_sparse # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

f = lambda w: softmax_loss_naive(w, X_dev, Y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

# similar to SVM case, do another gradient check with regularization
loss, grad = softmax_loss_naive(W, X_dev, Y_dev, 5e1)
f = lambda w: softmax_loss_naive(w, X_dev, Y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

########################################################################################################################

# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.

import time

tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

from classifiers.softmax_loss import softmax_loss_vectorized
#from softmax_loss import softmax_loss_vectorized # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, Y_dev, 0.000005)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)

#######################################################################################################################

# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.

from classifiers.linear_classifier import LinearSoftmax
#from linear_classifier import LinearSoftmax # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

results = {}
best_val = -1
best_softmax = None

learning_rates = [3e-7, 5e-8, 9e-8]       #[1e-7, 5e-5]
regularization_strengths = [3e5, 5e5]     #[2.5e4, 5e4]

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################
grid_search = [(lr, reg) for lr in learning_rates for reg in regularization_strengths]
for lr, reg in grid_search:
    softmax_model = LinearSoftmax()
    softmax_model.train(X_train, Y_train, learning_rate=lr, reg=reg, num_iters=1000)
    Y_train_pred = softmax_model.predict(X_train)
    train_accuracy = np.mean(Y_train_pred == Y_train)
    Y_val_pred = softmax_model.predict(X_val)
    val_accuracy = np.mean(Y_val_pred == Y_val)
    results[(lr, reg)] = (train_accuracy, val_accuracy)
    if best_val < val_accuracy:
        best_val = val_accuracy
        best_softmax = softmax_model
################################################################################
#                              END OF YOUR CODE                                #
################################################################################

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e    reg %e    train accuracy: %f    val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

########################################################################################################################

# Evaluate the best softmax on test set
Y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(Y_test == Y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))

########################################################################################################################

# Visualize the learned weights for each class
w = best_softmax.W[:-1, :]    # strip out the bias
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])

########################################################################################################################
