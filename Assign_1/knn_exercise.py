#FILE NAME: knn_exercise.py    #########################################################################################
# k-Nearest Neighbor (kNN) exercise
# Complete and hand in this code completed (including its outputs and any supporting code outside of it)
#
# The kNN classifier consists of two stages:
#
#   During training, the classifier takes the training data and simply remembers it
#   During testing,  kNN classifies every test image by comparing to all training images and
#                    transfering the labels of the k most similar training examples
# The value of k is cross-validated
# In this exercise you will implement these steps and understand the basic Image Classification pipeline,
# cross-validation, and gain proficiency in writing efficient, vectorized code.
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from utilities.load_cifar10 import load_CIFAR10
#from load_cifar10 import load_CIFAR10 # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

#######################################################################################################################

# Load the raw CIFAR-10 data.
# Change the following path to the directory that your "cifar-10-batches-py" file exists

cifar10_dir = 'D:\\safe\\university\\arshad\\deep\\cifar-10-batches-py'

# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
   del X_train, Y_train
   del X_test, Y_test
   print('Clear previously loaded data.')
except:
   pass

X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', Y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)

#######################################################################################################################

# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(classes)
samples_per_class = 7
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(Y_train == y)  #find train data indexes with class y
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()

#######################################################################################################################

# Subsample the data for more efficient code execution in this exercise
num_training = 500   #5000
mask = list(range(num_training))
X_train = X_train[mask]
Y_train = Y_train[mask]

num_test = 50   #500
mask = list(range(num_test))
X_test = X_test[mask]
Y_test = Y_test[mask]

#######################################################################################################################

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)

#######################################################################################################################

from classifiers.knn_classifier import KNearestNeighbor
#from knn_classifier import KNearestNeighbor # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
knn_model = KNearestNeighbor()
knn_model.train(X_train, Y_train)

#######################################################################################################################

# open classifiers/knn_classifier.py and implement the function "compute_distances_two_loops" that
# uses a (very inefficient) double loop over all pairs of (test, train) examples and
# computes the distance matrix one element at a time.

# Test your implementation:
dists = knn_model.compute_distances_two_loops(X_test)
print(dists.shape)

#######################################################################################################################

# We can visualize the distance matrix: each row is for a single test example and
# its distances to all training examples
plt.imshow(dists, interpolation='none')
plt.show()

#######################################################################################################################

# Now implement the function "predict_labels" in classifiers/knn_classifier.py
# and run the code below: We use k = 1 (which is Nearest Neighbor).
Y_test_pred = knn_model.predict_labels(dists, k=1)

# Compute and print the fraction of correctly predicted examples
num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('For k=1 we got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# You should expect to see approximately 27% accuracy.

#######################################################################################################################

# Now lets try out a larger k, say k = 5:
Y_test_pred = knn_model.predict_labels(dists, k=5)
num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('For k=5 we got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

# You should expect to see a slightly better performance than with k = 1.

#######################################################################################################################

# Now lets speed up distance matrix computation by using partial vectorization
# with one loop. Implement the function "compute_distances_one_loop" in
# classifiers/knn_classifier.py and run the code below:
dists_one = knn_model.compute_distances_one_loop(X_test)

# To ensure that our vectorized implementation is correct, we make sure that it
# agrees with the naive implementation. There are many ways to decide whether
# two matrices are similar; one of the simplest is the Frobenius norm. In case
# you haven't seen it before, the Frobenius norm of two matrices is the square
# root of the squared sum of differences of all elements; in other words, reshape
# the matrices into vectors and compute the Euclidean distance between them.
difference = np.linalg.norm(dists - dists_one, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

#######################################################################################################################

# Now implement the fully vectorized version inside "compute_distances_no_loops" function
# and run the code
dists_two = knn_model.compute_distances_no_loops(X_test)

# check that the distance matrix agrees with the one we computed before:
difference = np.linalg.norm(dists - dists_two, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

#######################################################################################################################

# Let's compare how fast the implementations are
def time_function(f, *args):
    #Call a function f with its args and return the time (in seconds) that it took to execute.
    import time
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

two_loop_time = time_function(knn_model.compute_distances_two_loops, X_test)
print('Two loop version took %f seconds' % two_loop_time)

one_loop_time = time_function(knn_model.compute_distances_one_loop, X_test)
print('One loop version took %f seconds' % one_loop_time)

no_loop_time = time_function(knn_model.compute_distances_no_loops, X_test)
print('No loop version took %f seconds' % no_loop_time)

# you should see significantly faster performance with the fully vectorized implementation

#######################################################################################################################
# Cross-validation
# We have implemented the k-Nearest Neighbor classifier but we set the value k = 5 arbitrarily.
# We will now determine the best value of this hyperparameter with cross-validation.
#######################################################################################################################

num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
Y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# Y_train_folds should each be lists of length num_folds, where                #
# Y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
X_train_folds = np.array_split(X_train, num_folds)
Y_train_folds = np.array_split(Y_train, num_folds)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# We define "k_to_accuracies" as a dictionary
# holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}

################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
for k in k_choices:
    k_to_accuracies[k] = []
    for j in range(num_folds):
        # We first prepare the training data for current fold. We use List Comprehension since it's easy to write,
        # less headaches while implementing, and, it looks better! but, complicated.
        X_train_fold = np.concatenate([fold for i, fold in enumerate(X_train_folds) if j != i])
        Y_train_fold = np.concatenate([fold for i, fold in enumerate(Y_train_folds) if j != i])

        # Then, we use K-Nearest Neighbor algorithm:
        knn_model.train(X_train_fold, Y_train_fold)
        Y_pred_fold = knn_model.predict(X_train_folds[j], k=k)

        # At the end, we compute the accuracy for current fold:
        num_correct = np.sum(Y_pred_fold == Y_train_folds[j])
        accuracy = float(num_correct) / X_train_folds[j].shape[0]
        k_to_accuracies[k].append(accuracy)
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

#######################################################################################################################

# plot the raw observations
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()

#######################################################################################################################

# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.

best_k = 1    # this is for num_train=500 and num_test=50

knn_model = KNearestNeighbor()
knn_model.train(X_train, Y_train)
Y_test_pred = knn_model.predict(X_test, k=best_k)

# Compute and display the accuracy
num_correct = np.sum(Y_test_pred == Y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))
