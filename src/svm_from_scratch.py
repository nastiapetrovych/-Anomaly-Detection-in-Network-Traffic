"""
SVM from scratch implementation
"""

import matplotlib.pyplot as plt
import numpy as np


class SVM:
    def __init__(self, kernel='linear', learning_rate=0.001, lambda_parameter=0.01, iterations=1000, degree=3, gamma=0.1):
        self.kernel = kernel
        self.lr = learning_rate
        self.lambda_parameter = lambda_parameter
        self.iterations = iterations
        self.degree = degree
        self.gamma = gamma
        self.X_train = None
        self.w = None
        self.b = None

    def fit(self, X, y):
        self.X_train = X
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_samples)
        self.b = 0

        for i in range(self.iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(self.w, self.kernel_function(X, x_i)) - self.b) >= 1
                if condition:
                    self.w[idx] -= self.lr * (2 * self.lambda_parameter * self.w[idx])
                else:
                    self.w[idx] -= self.lr * (2 * self.lambda_parameter * self.w[idx] - y_[idx])
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(self.w, self.kernel_function(self.X_train, X).T) - self.b
        return np.sign(approx)

    def kernel_function(self, X, x):
        if self.kernel == 'linear':
            return np.dot(X, x.T)
        elif self.kernel == 'polynomial':
            return (1 + np.dot(X, x.T)) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(X - x, axis=1) ** 2)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(X, x.T) + self.degree)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")

    def visualize_results(self, X_test_pca, y_pred_custom, kernel):
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_custom, cmap='coolwarm', s=20)
        plt.title(f'SVM Predictions with {kernel} kernel (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Prediction')
        plt.savefig(f'result_{kernel}.png')
        plt.show()
