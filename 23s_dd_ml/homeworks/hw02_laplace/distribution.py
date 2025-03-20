import numpy as np
from scipy.constants import value


class LaplaceDistribution:
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        ####
        n = len(x)
        loc = np.median(x)
        sum = np.sum(np.abs(x - loc))
        return sum / n

    def __init__(self, features):
        '''
        Args:
            feature: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        x = np.array(features)

        if x.ndim == 1:
            self.loc = np.median(features)
            self.scale = self.mean_abs_deviation_from_median(features)
        elif x.ndim == 2:
            self.loc = []  # YOUR CODE HERE
            self.scale = []
            for i in range(features.shape[1]):
                self.loc.append(np.median(features[:, i]))
                self.scale.append(self.mean_abs_deviation_from_median(features[:, i]))
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        x = np.array(values)

        if x.ndim == 1:
            logpdf_val = []
            for i in range(x.shape[0]):
                return logpdf_val.append(-np.log(1 / (2 * self.scale)) -np.abs(values[i] - self.loc)/ self.scale)
        elif x.ndim == 2:
            logpdf_val = []
            for i in range(x.shape[0]):
                logpdf_val1 = []
                for j in range(2):
                    logpdf_val1.append(-np.log(2 * self.scale[j]) - np.abs(values[i, j] - self.loc[j])/ self.scale[j])
                logpdf_val.append(logpdf_val1)
            return logpdf_val
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))
