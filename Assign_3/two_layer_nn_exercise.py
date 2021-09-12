#FILE NAME: two_layer_nn_exercise.py  ##################################################################################
# Implementing a Neural Network
# In this exercise we will develop a neural network with fully-connected layers to perform classification,
# and test it out on the CIFAR-10 dataset.
########################################################################################################################
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import random
from classifiers.neural_net_classifier import TwoLayerNet
#from neural_net_classifier import TwoLayerNet # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

########################################################################################################################
# We will use the class TwoLayerNet in the file classifiers/neural_net_classifier.py
# to represent instances of our network. The network parameters are stored in the instance variable self.params
# where keys are string parameter names and values are numpy arrays.
# Below, we initialize toy data and a toy model that we will use to develop your implementation.

# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments (reproducibility).

input_size = 4      # D: input data dimension
hidden_size = 10    # H
num_classes = 3     # C
num_inputs = 5      # N

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)   # N * D
    Y = np.array([0, 1, 2, 2, 1])
    return X, Y

net = init_toy_model()
X, Y = init_toy_data()

########################################################################################################################
# FORWARD PASS: COMPUTE SCORES
# Open the file classifiers/neural_net_classifier.py and look at the method TwoLayerNet.loss.
# This function is very similar to the loss functions you have written for the SVM and Softmax exercises:
# It takes the data and weights and computes the class scores, the loss, and the gradients on the parameters.
#
# Implement the first part of the forward pass which uses the weights and biases to compute the scores for all inputs.

scores = net.loss(X)         # without Y parameter loss function only returns scores
print('Your scores:')
print(scores)
print()
print('correct scores:')

correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. We get < 1e-7
print('Sum of differences between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))
print()

########################################################################################################################
# FORWARD PASS: COMPUTE LOSS
# In the same function (TwoLayerNet.loss),
# implement the second part that computes the data and regularizaion loss.

loss, _ = net.loss(X, Y, reg=0.05)
correct_loss = 1.30378789133

# The difference should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.abs(loss - correct_loss))
print()

########################################################################################################################
# BACKWARD PASS
# Implement the rest of the function TwoLayerNet.loss.
# This will compute the gradient of the loss with respect to the variables W1, b1, W2, and b2.
# Now that you (hopefully!) have a correctly implemented forward pass,
# you can debug your backward pass using a numeric gradient check:

from utilities.gradient_check import eval_numerical_gradient
#from gradient_check import eval_numerical_gradient # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, Y, reg=0.05)

# these should all be less than 1e-8 or so.
for param_name in grads:
    f = lambda W: net.loss(X, Y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
print()

########################################################################################################################
# TRAIN THE NETWORK
# To train the network we will use stochastic gradient descent (SGD), similar to the SVM and Softmax classifiers.
# Look at the function TwoLayerNet.train and fill in the missing sections to implement the training procedure.
# This should be very similar to the training procedure you used for the SVM and Softmax classifiers.
# You will also have to implement TwoLayerNet.predict, as the training process periodically performs prediction
# to keep track of accuracy over time while the network trains.
#
# Once you have implemented the method, run the code below to train a two-layer network on toy data.
# You should achieve a training loss less than 0.2.

net = init_toy_model()
stats = net.train(X, Y, X, Y,       # in toy example X_val=X-train
                  learning_rate=1e-1, reg=5e-6,
                  num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])
print()

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

########################################################################################################################
# LOAD CIFAR10 DATA
# Now that you have implemented a two-layer network that passes gradient checks and works on toy data,
# it's time to load up our favorite CIFAR-10 data so we can use it to train a classifier on a real dataset.

from utilities.load_cifar10 import load_CIFAR10
#from load_cifar10 import load_CIFAR10 # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'D:\\safe\\university\\arshad\\deep\\cifar-10-batches-py'

    X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    Y_val = Y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    Y_train = Y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    Y_test = Y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
try:
    del X_train, Y_train
    del X_test, Y_test
    print('Clear previously loaded data.')
except:
    pass

# Invoke the above function to get our data.
X_train, Y_train, X_val, Y_val, X_test, Y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', Y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', Y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)
print()

########################################################################################################################
# TRAIN A NETWORK
# To train our network we will use SGD. In addition, we will adjust the learning rate with
# an exponential learning rate schedule as optimization proceeds;
# after each epoch, we will reduce the learning rate by multiplying it by a decay rate.

input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, Y_train, X_val, Y_val,
                  num_iters=1000, batch_size=200,
                  learning_rate=1e-4, learning_rate_decay=0.95,
                  reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == Y_val).mean()
print('Final validation accuracy: ', val_acc)
print()

########################################################################################################################
# DEBUG THE TRAINING
# With the default parameters we provided above, you should get a validation accuracy of about 0.29
# on the validation set. This isn't very good.
#
# One strategy for getting insight into what's wrong is to plot the loss function and the accuracies
# on the training and validation sets during optimization.
#
# Another strategy is to visualize the weights that were learned in the first layer of the network.
# In most neural networks trained on visual data,
# the first layer weights typically show some visible structure when visualized.

# Plot the loss function and train / validation accuracies

def plot_loss_acc(stats):
    plt.subplot(2, 1, 1)
    plt.plot(stats['loss_history'])
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(stats['train_acc_history'], label='train')
    plt.plot(stats['val_acc_history'], label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.legend()
    plt.show()

plot_loss_acc(stats)

########################################################################################################################
from utilities.vis_utils import visualize_grid
#from vis_utils import visualize_grid # Since I don't use PyCharm, I had to use this line; the only modification out of bounds.

# Visualize the weights of the network

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)   # ?? -1 becomes H=50
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)

########################################################################################################################
# TUNE YOUR HYPERPARAMETERs
# What's wrong?. Looking at the visualizations above, we see that the loss is decreasing more or less linearly,
# which seems to suggest that the learning rate may be too low. Moreover, there is no gap between
# the training and validation accuracy, suggesting that the model we used has low capacity,
# and that we should increase its size. On the other hand, with a very large model we would expect to see
# more overfitting, which would manifest itself as a very large gap between the training and validation accuracy.
#
# TUNING. Tuning the hyperparameters and developing intuition for how they affect the final performance is
# a large part of using Neural Networks, so we want you to get a lot of practice.
# Below, you should experiment with different values of the various hyperparameters, including
# hidden layer size, learning rate, numer of training epochs, and regularization strength. You might also consider
# tuning the learning rate decay, but you should be able to get good performance using the default value.
#
# APPROXIMATE RESULTS. You should be aim to achieve a classification accuracy of greater than 48% on the validation set.
# Our best network gets over 52% on the validation set.
#
# EXPERIMENT. Your goal in this exercise is to get as good of a result on CIFAR-10 as you can,
# with a fully-connected Neural Network. Feel free implement your own techniques
# (e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).

best_net = None # store the best model into this

#################################################################################
# TODO: Tune hyperparameters using the validation set.                          #
# Store your best trained  model in best_net.                                   #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did in the previous exercises and                       #
# on the slides 73-75 of Lecture05_Training NN_1 (random search).               #
#################################################################################
input_size = 32 * 32 * 3
num_classes = 10

### How I did Random Search for best value for Hyperparmeters...
##hidden_sizes = [32, 64, 128, 256]
##learning_rates = [1e-3, 1e-4, 1e-5]
##learning_rate_decays = [0.90, 0.95, 0.97, 0.99]
##regularization_strengths = [0.25, 0.5, 1.0, 1.5]
##
##best_val = -1
##results = {}
##
##for hs in hidden_sizes:
##    for lr in learning_rates:
##        for rs in regularization_strenghts:
##            for decay in learning_rate_decays:
##                net = TwoLayerNet(input_size, hs, num_classes)
##                stats = net.train(X_train, Y_train, X_val, Y_val, num_iters=1000, batch_size=200,
##                                  learning_rate=lr, learning_rate_decay=decay, reg=rs, verbose=False)
##
##                train_accuracy = (net.predict(X_train) == Y_train).mean()
##                val_accuracy = (net.predict(X_val) == Y_val).mean()
##
##                results[hs, lr, rs, decay] = train_accuracy, val_accuracy
##
##                if val_accuracy > best_val:
##                    best_val = val_accuracy
##                    best_net = net
##                    best_stats = stats
##
##for hs, lr, reg, decay  in sorted(results):
##    train_accuracy, val_accuracy = results[(hs, lr, reg, decay)]
##    print('hs %e lr %e reg %e decay %e train accuracy: %f val accuracy: %f' % (hs, lr, reg, decay, train_accuracy, val_accuracy))
##
##print('Best validation accuracy achieved during cross-validation: %f' % best_val)


# The best that I could achieve...
hidden_size = 256
learning_rate = 1e-3
learning_rate_decay = 0.95
regularization_strength = 0.25

# Create the model.
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the model.
stats = net.train(X_train, Y_train, X_val, Y_val, num_iters=10000, batch_size=200, learning_rate=learning_rate,
                  learning_rate_decay=learning_rate_decay, reg=regularization_strength, verbose=True)

# Predict on the validation-set.
val_accuracy = (net.predict(X_val) == Y_val).mean()
print('Validation Accuracy: ', val_accuracy)

best_net = net
#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################


########################################################################################################################
# visualize the weights of the best network
show_net_weights(best_net)

########################################################################################################################
# RUN ON THE TEST SET
# When you are done experimenting, you should evaluate
# your final trained network on the test set; you should get above 48%.

test_acc = (best_net.predict(X_test) == Y_test).mean()
print('Test accuracy: ', test_acc)
