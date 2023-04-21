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

    # Normalisation

    means_val = np.mean(xtrain, axis=0, keepdims=True)
    stds_val = np.std(xtrain, keepdims=True)

    xtrain = (xtrain - means_val) / stds_val
    xtest = (xtest - means_val) / stds_val

    cross_xtrain = np.copy(xtrain)
    cross_ytrain = np.copy(ytrain)

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
    else:
        method_obj = KMeans(K=20, max_iters=10)

    
    ## 4. Train and evaluate the method

    # Cross validation

    if not args.test:
        bestAccuracy = 0
        bestLr = 0
        learningRateRange = [0.00000000001 + (0.001-0.00000000001)/250* x for x in range(0, 250)]
        accuracies = []
        if args.method == "logistic_regression":
            for lr_temp in learningRateRange:
                method_obj_temp = LogisticRegression(lr = lr_temp, max_iters = args.max_iters)
                preds_train = method_obj_temp.fit(xtrain, ytrain)
                preds_valid = method_obj_temp.predict(xvalid)
                accuracy = accuracy_fn(preds_valid, yvalid)
                accuracies.append(accuracy)
                if accuracy > bestAccuracy:
                    print(f"\nNew best validation set accuracy with lr = {lr_temp:.6f}: accuracy = {accuracy:.3f}")
                    bestAccuracy = accuracy
                    bestLr = lr_temp
                else:
                    pass
                    #print(f"\nValidation set accuracy with lr = {lr_temp}: accuracy = {accuracy:.3f}")
            method_obj = LogisticRegression(lr = bestLr, max_iters = args.max_iters)
            preds_train = method_obj.fit(xtrain, ytrain)

            axes = plt.gca()
            axes.set_ylim(0, 100)
            plt.plot(learningRateRange, accuracies)
            plt.xlabel("Learning rate")
            plt.ylabel("Accuracy")
            plt.title("Accuracy as a function of the learning rate for the logistic regression")
            plt.show()
        elif args.method == "svm":
            top_param_svm_linear = [1]
            best_acc_linear = KFold_cross_validation_SVM(X=cross_xtrain, Y=cross_ytrain, K=3, c=1, kernel='linear')

            top_param_svm_poly = [1, 1, 0, 0]  # c, gamma, degree, coef0
            best_acc_poly = KFold_cross_validation_SVM(X=cross_xtrain, Y=cross_ytrain, K=3,
                                                       c=1, kernel='poly', gamma=1, degree=0)

            top_param_svm_rbf = [1, 1]  # c, gamma
            best_acc_rbf = KFold_cross_validation_SVM(X=cross_xtrain, Y=cross_ytrain, K=3, c=1, kernel='rbf', gamma=1)

            range_c = [1, 50, 100, 500, 1000, 10000]
            range_gamma = [0.001, 0.01, 0.1, 1, 50]
            range_degree = np.arange(7)
            range_coef0 = np.arange(1)

            for c in range_c:
                acc_val = KFold_cross_validation_SVM(X=cross_xtrain, Y=cross_ytrain, K=3, c=c, kernel='linear')
                print(f"Accuracy : {acc_val} (C = {c}, kernel = linear)\n")
                if acc_val > best_acc_linear:
                    top_param_svm_linear = [c]
                    best_acc_linear = acc_val
                    print(f"We have a new best accuracy : {acc_val} (C = {c}, kernel = linear)\n")

            for c in range_c:
                for gamma in range_gamma:
                    for degree in range_degree:
                        for coef0 in range_coef0:
                            acc_val = KFold_cross_validation_SVM(X=cross_xtrain, Y=cross_ytrain, K=3, c=c,
                                                                 kernel='poly', gamma=gamma, degree=degree, coef0=coef0)
                            print(f"Accuracy : {acc_val} "
                                  f"(C = {c}, kernel = poly, gamma = {gamma}, degree = {degree}, coef0 = {coef0})\n")
                            if acc_val > best_acc_poly:
                                print(f"We have a new best accuracy : {acc_val} "
                                      f"(C = {c}, kernel = poly, gamma = {gamma}, degree = {degree}, coef0 = {coef0})\n")
                                top_param_svm_poly = [c, gamma, degree, coef0]
                                best_acc_poly = acc_val

            for c in range_c:
                for gamma in range_gamma:
                    acc_val = KFold_cross_validation_SVM(X=cross_xtrain, Y=cross_ytrain, K=3, c=c,
                                                         kernel='rbf', gamma=gamma)
                    print(f"Accuracy : {acc_val} "
                          f"(C = {c}, kernel = rbf, gamma = {gamma})\n")
                    if acc_val > best_acc_rbf:
                        print(f"We have a new best accuracy : {acc_val} "
                              f"(C = {c}, kernel = rbf, gamma = {gamma})\n")
                        top_param_svm_rbf = [c, gamma]
                        best_acc_rbf = acc_val

            print(f"The best accuracy for SVM with a linear kernel was {best_acc_linear} "
                  f"(reached for C = {top_param_svm_linear[0]})\n")

            print(f"The best accuracy for SVM with a polynomial kernel was {best_acc_poly} "
                  f"(reached for C = {top_param_svm_poly[0]}, gamma = {top_param_svm_poly[1]}, "
                  f"degree = {top_param_svm_poly[2]}, coef0 = {top_param_svm_poly[3]})\n")

            print(f"The best accuracy for SVM with RBF kernel was {best_acc_rbf} "
                  f"(reached for C = {top_param_svm_rbf[0]}, gamma = {top_param_svm_rbf[1]})\n")

            c_default = 500

            c_range = np.arange(0.001, 10, 0.2)
            acc_linear = []

            degree_range = np.arange(13)
            acc_poly = []

            gamma_range = [0.001 * x for x in np.arange(11)]
            acc_rbf = []

            for c in c_range:
                print(c)
                svm_linear = SVM(c, 'linear')
                svm_linear.fit(xtrain, ytrain)
                acc_linear.append(accuracy_fn(svm_linear.predict(xtest), ytest))

            print("Linear kernel finished...")

            for d in degree_range:
                print(d)
                svm_poly = SVM(C=c_default, kernel='poly', degree=d)
                svm_poly.fit(xtrain, ytrain)
                acc_poly.append(accuracy_fn(svm_poly.predict(xtest), ytest))

            print("Polynomial kernel finished...")

            for g in gamma_range:
                svm_rbf = SVM(C=c_default, kernel='rbf', gamma=g)
                svm_rbf.fit(xtrain, ytrain)
                acc_rbf.append(accuracy_fn(svm_rbf.predict(xtest), ytest))

            print("RBF kernel finished !")

            # plot linear
            plt.plot(c_range, acc_linear)
            plt.title("Accuracy as a function of C for linear kernel")
            plt.xlabel("C (with gamma=1, degree=1 and coef0=0)")
            plt.ylabel("Accuracy (in percentage)")
            plt.show()
            # plot poly
            # plt.plot(degree_range, acc_poly)
            # plt.title("Accuracy as a function of the degree for polynomial kernel")
            # plt.xlabel("Degree (with C=500, gamma=1, coef0=0)")
            # plt.ylabel("Accuracy (in percentage)")
            # plt.show()
            # plot rbf
            # plt.plot(gamma_range, acc_rbf)
            # plt.title("Accuracy as a function of gamma for RBF kernel")
            # plt.xlabel("Gamma (with C=500, coef0=0)")
            # plt.ylabel("Accuracy (in percentage)")
            # plt.show()
        else: 
            
    
            k_axis = []
            accuracy_axis = []
            better_accuracy = 0
            better_k = 0
            for k in range(1, 100):
                method_obj.K = k
                preds_train = method_obj.fit(xtrain, ytrain)
                preds = method_obj.predict(xvalid)
                accuracy_with_a_certain_k = accuracy_fn(method_obj.predict(xvalid), yvalid)
                k_axis.append(k)
                accuracy_axis.append(accuracy_with_a_certain_k)
                acc = accuracy_fn(preds_train, ytrain)
                macrof1 = macrof1_fn(preds_train, ytrain)
                print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

                acc = accuracy_fn(preds, yvalid)
                macrof1 = macrof1_fn(preds, yvalid)
                print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
                if (accuracy_with_a_certain_k > better_accuracy):
                    better_accuracy = accuracy_with_a_certain_k
                    print(better_accuracy)
                    better_k = k
            print(better_k)
            plt.plot(k_axis, accuracy_axis)
            plt.xlabel("K")
            plt.ylabel("accuracy")
            plt.title("accuracy on validation data in function of hyperparameter K")
            plt.show()

            method_obj.K = better_k
        
            preds_train = method_obj.fit(xtrain, ytrain)
    else :
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

def KFold_cross_validation_SVM(X, Y, K, c, kernel, gamma=1, degree=0, coef0=0):
    '''
    K-Fold Cross validation function for K-NN
    Inputs:
        X : training data, shape (NxD)
        Y: training labels, shape (N,)
        K: number of folds (K in K-fold)
    Returns:
        Average validation accuracy for the selected hyperparameters.
    '''
    N = X.shape[0]

    accuracies = []  # list of accuracies
    for fold_ind in range(K):
        # Split the data into training and validation folds:

        # all the indices of the training dataset
        all_ind = np.arange(N)
        split_size = N // K

        # Indices of the validation and training examples
        val_ind = all_ind[fold_ind * split_size: (fold_ind + 1) * split_size]
        train_ind = np.setdiff1d(all_ind, val_ind)

        X_train_fold = X[train_ind, :]
        Y_train_fold = Y[train_ind]
        X_val_fold = X[val_ind, :]
        Y_val_fold = Y[val_ind]

        # Run KNN using the data folds you found above.
        model = SVM(C=c, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
        model.fit(X_train_fold, Y_train_fold)
        Y_val_fold_pred = model.predict(X_val_fold)
        acc = accuracy_fn(Y_val_fold_pred, Y_val_fold)
        accuracies.append(acc)

    # Find the average validation accuracy over K:
    ave_acc = np.sum(accuracies) / K
    return ave_acc


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