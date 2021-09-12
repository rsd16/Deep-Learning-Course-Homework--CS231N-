#FILE NAME: softmax_loss.py

import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, Y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
    unorm_log_probs = X[i].dot(W)
    unorm_log_probs -= np.max(unorm_log_probs)
    exp_sum = np.sum(np.exp(unorm_log_probs))
    softmax = np.exp(unorm_log_probs[Y[i]]) / exp_sum
    loss -= np.log(softmax)
    scores = np.exp(unorm_log_probs) / exp_sum
    for j in range(num_classes):
      if j == Y[i]:
        dscore = scores[j] - 1
      else:
        dscore = scores[j]

      dW[:, j] += dscore * X[i]

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, Y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    num_classes = W.shape[1]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    unorm_log_probs = X.dot(W)
    unorm_log_probs -= np.max(unorm_log_probs)
    scores_exp = np.exp(unorm_log_probs)
    loss += -np.sum(np.log(scores_exp[range(num_train), Y])) + np.sum(np.log(np.sum(scores_exp, axis=1)))
    S = scores_exp / np.sum(scores_exp, axis=1)[:, None]
    S_corr = S - np.equal(np.arange(num_classes), Y[:, None])
    dW += np.dot(X.T, S_corr)

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
