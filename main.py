import argparse

import numpy as np 
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn
 
def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)


    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.


    # TODO: Normalization of the data


    # Suffle of the training data
    indices = np.random.permutation(xtrain.shape[0])
    xtrain, ytrain = xtrain[indices, :], ytrain[indices]


    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ratio = 0.7
        N = xtrain.shape[0]
        limit = (int) (ratio * N)
        
        xtrain, xvalid = xtrain[:limit, :], xtrain[limit:, :]
        ytrain, yvalid = ytrain[:limit], ytrain[limit:]

        
    
    # Dimensionality reduction (FOR MS2!)
    if args.use_pca:
        raise NotImplementedError("This will be useful for MS2.")
    

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    
    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr = args.lr, max_iters = args.max_iters)

    elif args.method == "svm":
        method_obj = SVM(C=args.svm_c, kernel=args.svm_kernel, gamma=args.svm_gamma)
        pass

    
    ## 4. Train and evaluate the method

    # Cross validation

    #python main.py --data dataset_HASYv2  --method logistic_regression

    if not args.test:
        bestAccuracy = 0
        bestLr = 0
        learningRateRange = [0.0000001 + 0.00001*x for x in range(0, 90)]
        accuracies = []
        if args.method == "logistic_regression":
            for lr_temp in learningRateRange:
                method_obj_temp = LogisticRegression(lr = lr_temp, max_iters = args.max_iters)
                preds_train = method_obj_temp.fit(xtrain, ytrain)
                preds_valid = method_obj_temp.predict(xvalid)
                accuracy = accuracy_fn(preds_valid, yvalid)
                accuracies.append(accuracy)
                if accuracy > bestAccuracy:
                    print(f"\nNew best validation set accuracy with lr = {lr_temp}: accuracy = {accuracy:.3f}")
                    bestAccuracy = accuracy
                    bestLr = lr_temp
                else:
                    print(f"\nValidation set accuracy with lr = {lr_temp}: accuracy = {accuracy:.3f}")
            method_obj = LogisticRegression(lr = bestLr, max_iters = args.max_iters)
            preds_train = method_obj.fit(xtrain, ytrain)

            axes = plt.gca()
            axes.set_ylim(0, 100)
            plt.plot(learningRateRange, accuracies)
            plt.xlabel("Learning rate")
            plt.ylabel("Accuracy")
            plt.title("Accuracy as a function of the learning rate for the logistic regression")
            plt.annotate(f"best accuracy = {bestAccuracy:.2f}", xy=(bestLr, bestAccuracy), xytext=(bestLr-0.00015, bestAccuracy+5),
                arrowprops=dict(facecolor='black', shrink=0.005),
                )
            plt.show()

        # Fit (:=train) the method on the training data
        

        # 
            

    # Predict on unseen data
    preds = method_obj.predict(xtest)


    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    # Feel free to add more arguments here if you need!

    # Arguments for MS2
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")

    # "args" will keep in memory the arguments and their value,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)

#def cross_validation():