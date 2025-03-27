import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Computes entropy of the provided distribution. Use log(value + eps) for numerical stability
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Entropy of the provided subset
    """
    EPS = 0.0005
    # YOUR CODE HERE
    y_all = np.zeros(y.shape[-1])
    for i in range(y.shape[0]):
        for j in range(len(y_all)):
            y_all[j] += y[i, j]

    y_all /= float(y.shape[0])
    sum = 0
    for i in range(y_all.shape[0]):
        sum += y_all[i] * np.log(y_all[i] + EPS)
    
    return  -sum
    
def gini(y):
    """
    Computes the Gini impurity of the provided distribution
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, n_classes)
        One-hot representation of class labels for corresponding subset
    
    Returns
    -------
    float
        Gini impurity of the provided subset
    """

    # YOUR CODE HERE
    y_all = np.zeros(y.shape[-1])
    for i in range(y.shape[0]):
        for j in range(len(y_all)):
            y_all[j] += y[i, j]

    y_all /= float(y.shape[0])
    y_all *= y_all
    # YOUR CODE HERE

    return 1 - np.sum(y_all)
    
def variance(y):
    """
    Computes the variance the provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Variance of the provided target vector
    """
    
    # YOUR CODE HERE
    R = y.shape[0]
    sum = 0
    y_mean = np.mean(y)
    for i in range(R):
        sum += (y[i] - y_mean) ** 2
    
    return sum / R

def mad_median(y):
    """
    Computes the mean absolute deviation from the median in the
    provided target values subset
    
    Parameters
    ----------
    y : np.array of type float with shape (n_objects, 1)
        Target values vector
    
    Returns
    -------
    float
        Mean absolute deviation from the median in the provided vector
    """

    # YOUR CODE HERE
    R = y.shape[0]
    sum = 0
    y_median = np.median(y)
    for i in range(R):
        sum += np.abs(y[i] - y_median)

    return sum / R


def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_one_hot[np.arange(len(y)), y.astype(int)[:, 0]] = 1.
    return y_one_hot


def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]


class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
        
        
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug

        
        
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Makes split of the provided data subset and target values using provided feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        (X_left, y_left) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j < threshold
        (X_right, y_right) : tuple of np.arrays of same type as input X_subset and y_subset
            Part of the providev subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        left_mask = X_subset[:, feature_index] < threshold
        right_mask = ~left_mask

        X_left = X_subset[left_mask]
        y_left = y_subset[left_mask]

        X_right = X_subset[right_mask]
        y_right = y_subset[right_mask]
        
        return (X_left, y_left), (X_right, y_right)

    def make_tree(self, X_subset, y_subset):
        """
        Recursively builds the tree

        Parameters
        ----------
        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                   (n_objects, 1) in regression
            One-hot representation of class labels or target values for corresponding subset

        Returns
        -------
        root_node : Node class instance
            Node of the root of the fitted tree
        """

        # YOUR CODE HERE
        return []
        #return new_node

    def fit(self, X, y):
        """
        Fit the model from scratch using the provided data

        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data to train on

        y : np.array of type int with shape (n_objects, 1) in classification
                   of type float with shape (n_objects, 1) in regression
            Column vector of class labels in classification or target values in regression

        """
        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        #self.root =  self.make_tree(X, y)
        self.root = entropy(y)

    def eval_G(self, y_all, y_left, y_right):

        Q = float(y_all.shape[0])
        L_Q = float(y_left.shape[0]) / Q
        R_Q = float(y_right.shape[0]) / Q

        if self.criterion_name == 'gini':
            return gini(y_all) - L_Q * gini(y_left) - R_Q * gini(y_right)
        elif self.criterion_name == 'entropy':
            return entropy(y_all) - L_Q * entropy(y_left) - R_Q * entropy(y_right)
        elif self.criterion_name == 'variance':
            y_all_decode = one_hot_decode(y_all)
            y_left_decode = one_hot_decode(y_left)
            y_right_decode = one_hot_decode(y_right)
            return variance(y_all_decode) - L_Q * variance(y_left_decode) - R_Q * variance(y_right_decode)
        elif self.criterion_name == 'mad_median':
            y_all_decode = one_hot_decode(y_all)
            y_left_decode = one_hot_decode(y_left)
            y_right_decode = one_hot_decode(y_right)
            return mad_median(y_all_decode) - L_Q * mad_median(y_left_decode) - R_Q * mad_median(y_right_decode)

        def choose_best_split(self, X_subset, y_subset):
            """
            Greedily select the best feature and best threshold w.r.t. selected criterion

            Parameters
            ----------
            X_subset : np.array of type float with shape (n_objects, n_features)
                Feature matrix representing the selected subset

            y_subset : np.array of type float with shape (n_objects, n_classes) in classification
                       (n_objects, 1) in regression
                One-hot representation of class labels or target values for corresponding subset

            Returns
            -------
            feature_index : int
                Index of feature to make split with

            threshold : float
                Threshold value to perform split

            """
            # YOUR CODE HERE
            feature_index = 1
            threshold = 1
            return feature_index, threshold

    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Split only target values into two subsets with specified feature and threshold
        
        Parameters
        ----------
        feature_index : int
            Index of feature to make split with

        threshold : float
            Threshold value to perform split

        X_subset : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the selected subset

        y_subset : np.array of type float with shape (n_objects, n_classes) in classification 
                   (n_objects, 1) in regression 
            One-hot representation of class labels for corresponding subset
        
        Returns
        -------
        y_left : np.array of type float with shape (n_objects_left, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j < threshold

        y_right : np.array of type float with shape (n_objects_right, n_classes) in classification 
                   (n_objects, 1) in regression 
            Part of the provided subset where selected feature x^j >= threshold
        """

        # YOUR CODE HERE
        left_mask = X_subset[:, feature_index] < threshold
        right_mask = ~left_mask

        y_left = y_subset[left_mask]
        y_right = y_subset[right_mask]
        return y_left, y_right
"""
    def predict(self, X):
        "" "
        Predict the target value or class label  the model from scratch using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted : np.array of type int with shape (n_objects, 1) in classification 
                   (n_objects, 1) in regression 
            Column vector of class labels in classification or target values in regression
        
        "" "

        # YOUR CODE HERE
        
        return y_predicted
        
    def predict_proba(self, X):
        "" "
        Only for classification
        Predict the class probabilities using the provided data
        
        Parameters
        ----------
        X : np.array of type float with shape (n_objects, n_features)
            Feature matrix representing the data the predictions should be provided for

        Returns
        -------
        y_predicted_probs : np.array of type float with shape (n_objects, n_classes)
            Probabilities of each class for the provided objects
        
        "" "
        assert self.classification, 'Available only for classification problem'

        # YOUR CODE HERE
        
        return y_predicted_probs
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification, make_regression, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error

RANDOM_STATE = 42

X = np.ones((4, 5), dtype=float) * np.arange(4)[:, None]
y = np.arange(4)[:, None] + np.asarray([0.2, -0.3, 0.1, 0.4])[:, None]
class_estimator = DecisionTree(max_depth=10, criterion_name='gini')

(X_l, y_l), (X_r, y_r) = class_estimator.make_split(1, 1., X, y)

print(np.array_equal(X[:1], X_l))
print(np.array_equal(X[1:], X_r))
print(np.array_equal(y[:1], y_l))
print(np.array_equal(y[1:], y_r))

digits_data = load_digits().data
digits_target = load_digits().target[:, None] # to make the targets consistent with our model interfaces
X_train, X_test, y_train, y_test = train_test_split(digits_data, digits_target, test_size=0.2, random_state=RANDOM_STATE)

class_estimator = DecisionTree(max_depth=10, criterion_name='gini')
class_estimator.fit(X_train, y_train)