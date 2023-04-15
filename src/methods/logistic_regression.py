import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        
        ## OTHER ARGUMENTS

         
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_label (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        self.C = get_n_classes(training_labels)
        self.D = training_data.shape[1]

        self.weights = self.logistic_regression_train_multi(training_data, 
                                                            label_to_onehot(training_labels, self.C), 
                                                            self.max_iters, self.lr)
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        return self.logistic_regression_predict_multi(test_data, self.weights)

    
    # METHODS FOR MULTI-CLASS LOGISTIC REGRESSION

    def f_softmax(self, data, W):
        expxw = np.exp(data@W)
        sum = np.reshape(np.sum(expxw, axis=1), (-1, 1))
        return expxw/sum
    

    def gradient_logistic_multi(self, data, labels, W):
        return np.transpose(data)@(self.f_softmax(data, W) - labels)
    

    def logistic_regression_predict_multi(self, data, W):
        proba = self.f_softmax(data, W)
        indices = np.argmax(proba, axis=1)
        return indices
    

    def logistic_regression_train_multi(self, data, labels, max_iters, lr):
        
        # Random initialization of the weights
        weights = np.random.normal(0, 0.1, (self.D, self.C))
        for it in range(max_iters):
            gradient = self.gradient_logistic_multi(data, labels, weights)
            weights -= lr*gradient

            predictions = self.logistic_regression_predict_multi(data, weights)
            if accuracy_fn(predictions, onehot_to_label(labels)) == 100:
                break
                
        return weights