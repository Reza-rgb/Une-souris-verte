import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

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
        ratio = 0.8
        N = xtrain.shape[0]
        limit = (int)(ratio * N)

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
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)

    elif args.method == "svm":
        method_obj = SVM(C=args.svm_c, kernel=args.svm_kernel, gamma=args.svm_gamma,
                         degree=args.svm_degree, coef0=args.svm_coef0)
        pass

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

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
    # Cross validation
    method_cross = method_obj
    if args.method == "logistic_regression":
        print("Not implemented yet")
    elif args.method == "svm":
        top_perf_svm = [args.svm_c, 'linear', args.svm_gamma, args.svm_degree, args.svm_coef0]
        best_acc = accuracy_fn(method_obj.predict(xvalid), yvalid)

        range_c = [1, 50, 100, 500, 1000, 10000]  # range(0.001, 10, 0.001)
        range_gamma = [1, 50, 100, 500, 1000]#range(1, 1001)
        range_degree = np.arange(15)#range(0, 101)
        range_coef0 = np.arange(5)#range(1, 1001) #naze

        for c in range_c:
            test_perf_svm = SVM(C=c, kernel='linear')
            test_perf_svm.fit(xtrain, ytrain)
            valid_acc = accuracy_fn(test_perf_svm.predict(xvalid), yvalid)
            print(f"Accuracy : {valid_acc} (C = {c}, kernel = linear)\n")
            if (valid_acc > best_acc):
                top_perf_svm = [c, 'linear', 0, 0, 0]
                print(f"We have a new best accuracy : {valid_acc} (C = {c}, kernel = linear)\n")


        #for c in range_c: (AROUND 10% ACCURACY => NOT INTERESTING !
        #    for gamma in range_gamma:
        #        test_perf_svm = SVM(C=c, kernel='rbf', gamma=gamma)
        #        test_perf_svm.fit(xtrain, ytrain)
        #        valid_acc = accuracy_fn(test_perf_svm.predict(xvalid), yvalid)
        #        print(f"Accuracy : {valid_acc} (C = {c}, kernel = rbf, gamma = {gamma})\n")
        #        if (valid_acc > best_acc):
        #            print(f"We have a new best accuracy : {valid_acc} (C = {c}, kernel = rbf, gamma = {gamma})\n")
        #            top_perf_svm = [c, 'rbf', gamma, 0, 0]
        #            best_acc = valid_acc

        for c in range_c:
            for gamma in range_gamma:
                for degree in range_degree:
                    for coef0 in range_coef0:
                        test_perf_svm = SVM(C=c, kernel='poly', gamma=gamma, degree=degree, coef0=coef0)
                        test_perf_svm.fit(xtrain, ytrain)
                        valid_acc = accuracy_fn(test_perf_svm.predict(xvalid), yvalid)
                        print(f"Accuracy : {valid_acc} "
                              f"(C = {c}, kernel = poly, gamma = {gamma}, degree = {degree}, coef0 = {coef0})\n")
                        if (valid_acc > best_acc):
                            print(f"We have a new best accuracy : {valid_acc} "
                                  f"(C = {c}, kernel = poly, gamma = {gamma}, degree = {degree}, coef0 = {coef0})\n")
                            top_perf_svm = [c, 'poly', gamma, degree, coef0]
                            best_acc = valid_acc
        pass


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str,
                        help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str,
                        help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear",
                        help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
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

# def cross_validation():