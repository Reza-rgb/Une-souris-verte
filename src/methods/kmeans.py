import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters
        self.distances = np.empty
        self.cluster_assignments = np.empty
        self.final_centers = np.empty
        self.cluster_center_label = np.empty

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        centers = self.init_centers(data)

        for i in range(max_iter):
            old_centers = centers.copy()
            self.compute_distance(data, old_centers)
            self.find_closest_cluster()
            centers = self.compute_centers(data)

            if np.all(centers == old_centers):
                break

        self.final_centers = centers
        self.compute_distance(data, old_centers)
        self.find_closest_cluster()

        return centers, self.cluster_assignments
    
    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        self.k_means(training_data)
        self.assign_labels_to_centers(self.final_centers, training_labels)
        
        return self.predict_with_centers(training_data, self.final_centers)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        self.compute_distance(test_data, self.final_centers)
        self.find_closest_cluster()
        pred_labels = self.predict_with_centers(test_data, self.final_centers)
        return pred_labels
    
    def init_centers(self, data):
        
        centers = np.random.permutation(data)[:self.K,]
        return centers

    def compute_distance(self, data, centers):

        N = data.shape[0]
        K = centers.shape[0]

        self.distances = np.zeros((N, K))

        for k in range(K):
            self.distances[:,k] = np.sqrt(np.sum((data - centers[k]) ** 2 , axis=1))


    def find_closest_cluster(self):

        self.cluster_assignments = np.argmin(self.distances, axis=1)
    
    def compute_centers(self, data):

        centers = np.zeros((self.K, np.shape(data)[1]))

        for k in range(self.K):
            mask = self.cluster_assignments == k
            centers[k] = np.sum(data[mask], axis=0) / np.sum(np.array(mask, dtype=int))

        return centers

    def assign_labels_to_centers(self, centers, true_labels):
        K = np.shape(centers)[0]
        self.cluster_center_label = np.zeros(K)
        for k in range(K):
            self.cluster_center_label[k] = np.argmax(np.bincount(true_labels[self.cluster_assignments == k]))

    def predict_with_centers(self, data, centers):
        N = np.shape(data)[0]
        new_labels = np.zeros((N,))
        self.compute_distance(data, centers)
        self.find_closest_cluster()
        for i in range(N):
            new_labels[i] = self.cluster_center_label[self.cluster_assignments[i]]

        return new_labels