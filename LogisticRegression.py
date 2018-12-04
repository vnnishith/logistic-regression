#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 22:46:30 2018

@author: nishithvn
"""
import numpy as np

class LogisticRegression:
    
    def __init__(self, max_iter=100000, learning_rate=.02,tolerance=1e-5):
        """
        Setting the class variables 
        defining the max_number_of_iterations,learning_rate & tolerance
        """
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tolerance = tolerance
                    
    def fit(self, X, y) :
        """
        Fit the coefficents using training data X, y. 
        X: 2D numpy array with numerical attributes as columns 
           and samples as rows
        y: numpy array of labels
        """
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        X = np.c_[np.ones(len(X)),X]
        # Initialise coefficients to zero
        self.theta = np.zeros((self.num_classes, X.shape[1]))
        # Track the number of iterations taken for each class
        self.num_iterations = np.zeros(self.num_classes, dtype=np.uint32)
        # Fit a logistic regression model for each class
        for index, actual_class in enumerate(self.classes) :
        # 1's for this class and 0's for all others
            modified_labels = 1*(y == actual_class)
            self.gradient_descent(X, modified_labels, index)
        self.intercept_ = self.theta[:,0]
        self.coef_ = self.theta[:,1:]
        
    def predict(self, X) :
        """Return numpy array of predicted classes for test set X"""
        X = np.c_[np.ones(len(X)),X]
        prob = self.h(0, X)
        for i in range(1, self.num_classes):
            prob = np.c_[prob, self.h(i, X)]
        most_probable_indices = np.argmax(prob, axis = 1)
        most_probable_classes = self.classes[most_probable_indices]
        return most_probable_classes
     
    def gradient_descent(self, X, y, index = 0) :
        """
        Fits binary logistic model for class given by index. 
        Stores the coefficients in self.theta.
        """
        while self.num_iterations[index] < self.max_iter:
            old_cost = self.cost(X, y, index)
            self.theta[index] -= self.learning_rate * self.grad_cost(X, y, index)
            new_cost = self.cost(X, y, index)
            # STOP if convergence is met.
            if abs(new_cost - old_cost) < self.tolerance:
                break
            self.num_iterations += 1
            
    def h(self, index, X) :
        """
        Hypothesis function 
        index: the class being fit
        """
        return 1.0/(1 + np.exp(-X.dot(self.theta[index])))
    
    def cost(self, X, y, index) :
        """
        cost function implementation
        J(θ)=1/m[∑y(i)log(hθ(x(i)))+(1−y(i))log(1−hθ(x(i)))]
        """
        h = self.h(index, X)
        return -(np.sum(y.dot(np.log(h)) + (1-y).dot(np.log(1-h))))/len(y)
    
    def grad_cost(self, X, y, index) :
        """Return the gradient of the cost function"""
        error = -(y - self.h(index, X))
        return error.dot(X)      