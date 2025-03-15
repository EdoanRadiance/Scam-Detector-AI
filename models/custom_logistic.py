import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib





def sigmoid(z):
    return 1/(1 + np.exp(-z))

class CustomLogisticRegression:
    def __init__(self, learning_rate=0.5, iterations = 1000, verbose=False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.verbose = verbose
        self.weights = None


    def Fit(self, X,Y):
        n_samples, n_features = X.shape
        X_bias = np.hstack((np.ones((n_samples, 1)), X))

        self.weights = np.zeros(X_bias.shape[1])

        for i in range(self.iterations):
            z = np.dot(X_bias, self.weights)
            y_hat = sigmoid(z)
            # The gradient for logistic regression (cross-entropy loss) is:
            # gradient = (1/n_samples) * X_bias^T · (y_hat - y)
            gradient = (1 / n_samples) * np.dot(X_bias.T, (y_hat - Y))
            # Update the weights using gradient descent.
            self.weights -= self.learning_rate * gradient

            if self.verbose and i % 100 == 0:
                    loss = -np.mean(Y * np.log(y_hat + 1e-10) + (1 - Y) * np.log(1 - y_hat + 1e-10))
                    print(f"Iteration {i}: Loss = {loss}")


    def predict_proba(self, X):
        """
        Predict the probabilities for input features X.
        
        Parameters:
        - X: A numpy array of shape (n_samples, n_features).
        
        Returns:
        - A numpy array of shape (n_samples,) with the predicted probabilities.
        """
        n_samples = X.shape[0]
        X_bias = np.hstack((np.ones((n_samples, 1)), X))
        z = np.dot(X_bias, self.weights)
        return sigmoid(z)



    def predict(self, X, threshold=0.3):
        """
        Predict binary labels for input features X.
        
        Parameters:
        - X: A numpy array of shape (n_samples, n_features).
        - threshold: Threshold to convert probabilities to binary labels.
        
        Returns:
        - A numpy array of shape (n_samples,) with binary predictions (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


