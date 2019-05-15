import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
  
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    pass
    XW1 = np.dot(X, W1) + b1
    Relu1 = np.maximum(XW1, np.zeros_like(XW1))
    scores = np.dot(Relu1, W2) + b2
    
    if y is None:
      return scores

    # Compute the loss
    loss = None
    pass
    scores = np.exp(scores.T-scores.max(axis=1)).T
    correct = scores[np.arange(scores.shape[0]), y]
    scores_total = np.sum(scores, axis = 1)
    loss = np.sum(-np.log(np.divide(correct, scores_total)))/scores.shape[0]
    loss += 0.5 * reg * (np.sum(W1*W1) + np.sum(W2*W2))
    
    # Backward pass: compute gradients
    grads = {}
    pass
    dW1 = np.zeros_like(W1)
    dW2 = np.zeros_like(W2)
    db1 = np.zeros_like(b1)
    db2 = np.zeros_like(b2)
   
    grad = (scores.T/scores_total)
    grad[y,np.arange(scores.shape[0])] += -1.0
    dW2 = np.dot(grad, Relu1).T/N + reg * W2

    db2 = np.sum(grad, axis = 1)/N
   
    dRelu1 = W2.dot(grad)
    dXW1 = (XW1>0) * dRelu1.T
    db1 = np.sum(dXW1, axis = 0)/N
    dW1 = X.T.dot(dXW1) + reg * W1
    grads['b2']=db2
    grads['W2']=dW2
    grads['b1']=db1
    grads['W1']=dW1
    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']

    for it in xrange(num_iters):
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      idx = np.random.choice(y.shape[0], batch_size)
      X_batch = X[idx]
      y_batch = y[idx]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      W1 -= learning_rate*grads['W1']
      W2 -= learning_rate*grads['W2']
      b1 -= learning_rate*grads['b1']
      b2 -= learning_rate*grads['b2']

      self.params['W1']=W1
      self.params['b1']=b1
      self.params['W2']=W2
      self.params['b2']=b2
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
	pass
    y_pred = np.argmax(self.loss(X), axis = 1)
    
    return y_pred



