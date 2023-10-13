from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import sys

'''
The key concept in CSP is to find a set of spatial filters (components) that optimally discriminate 
between the two classes. These filters are represented by the eigenvectors obtained in the 'fit' method. 
When you apply the CSP transformation to a new data sample in the 'transform' method, it projects the data 
onto these filters. The result is that the transformed data has enhanced features that maximize the 
differences in variances between the two classes. This makes it easier to classify the data based on 
the most discriminative spatial patterns.
'''

class CSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nb_components=4): 
        '''
        In practice, it's common to try a range of values for the number of components (e.g., from 2 to 10) 
        and evaluate their impact on classification accuracy.
        The choice of the optimal number of components may involve a trade-off between enhanced discriminative power 
        and computational efficiency.
        '''
        self.nb_components = nb_components
        self.filters = np.array([])

    def fit(self, x, y):
        class_labels = np.unique(y)
        if len(class_labels) != 2:
            print("CSPTransformer: Error: CSP is a binary classification method: there should be two class labels.", 
                  file=sys.stderr)
            exit()
        x_class1 = x[y == class_labels[0]] #take all values related to one class-label
        x_class2 = x[y == class_labels[1]]
        '''
        Get covariance matrices for each class.
        A covariance matrix is a square matrix giving the covariance between each element pair in a vector
        Covariance is the mean value of the product of the deviations of two variates from their respective means.
        A positive covariance indicates that both variates tend to be high or low at same time (similar direction)
        while a negative covariance indicates that if one variate is high the other will be low (opposite direction).
        '''
        cov1 = np.cov(x_class1, rowvar=False)
        cov2 = np.cov(x_class2, rowvar=False)
        '''
        Get the 'eigenvalues' and 'eigenvectors' by solving the 'generalized eigenvalue problem'
        The 'generalized eigenvalue problem' is a mathematical problem that arises in various fields 
        and is an extendion of the 'standard eigenvalue problem'.
        In the 'standard eigenvalue problem', you are given a square matrix A, and you want to find 
        scalars λ (eigenvalues) and corresponding vectors x (eigenvectors) that satisfy 
        the equation: A * x = λ * x. The 'eigenvalues' represent the scaling factors, and the 'eigenvectors' 
        represent the directions along which the matrix A scales or rotates.
        The 'generalized eigenvalue problem' extends the concept to two matrices, A and B. Given two square matrices 
        A and B, you want to find scalars λ (generalized eigenvalues) and corresponding 
        vectors x (generalized eigenvectors) that satisfy the equation: A * x = λ * B * x.
        Here we will use the 'eigenvectors' as transformation matrices that will maximize the variance of one class
        and minimize the variance of another class with the goal of maximizing the differences between the two classes.
        '''
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.pinv(np.add(cov1, cov2)), cov1))
        '''
        Now we will sort eigenvalues and corresponding eigenvectors in descending order. Thus sort the eigenvalues and 
        align the corresponding eigenvectors based on the eigenvalue magnitudes.
        This will allow us to select the first eigenvectors and use those as CSP filters. Because they come first
        they will be associated with the highest eigenvalue magnitudes and thus be the most discriminative.
        '''
        ascending_indices = np.argsort(eigenvalues)
        descending_indices = np.flip(ascending_indices)
        eigenvalues = eigenvalues[descending_indices]
        eigenvectors = eigenvectors[:, descending_indices] #reorder the columns (eigenvectors) of the eigenvectors matrix
        self.filters = eigenvectors[:, :self.nb_components]
        return self

    def transform(self, x):
        if self.filters.size == 0:
            print("CSPTransformer: Error: use the 'fit' method to find the filters before using 'transform' method", 
                  file=sys.stderr)
            exit()
        x_csp = np.dot(x, self.filters)
        return x_csp
