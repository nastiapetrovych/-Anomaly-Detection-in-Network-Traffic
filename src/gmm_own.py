import numpy as np
from scipy.stats import multivariate_normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

import pandas as pd
from data_preparation import DataPreprocessor
from sklearn.model_selection import train_test_split
class CustomKMeans:
    def __init__(self, n_clusters=2, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None

    def fit(self, X):
        if self.random_state:
            np.random.seed(self.random_state)
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]
        
        for _ in range(self.max_iter):
            old_centers = np.copy(self.cluster_centers_)
            distances = np.sqrt(((X[:, np.newaxis] - self.cluster_centers_)**2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            
            for i in range(self.n_clusters):
                if np.any(labels == i):
                    self.cluster_centers_[i] = X[labels == i].mean(axis=0)
            
            if np.all(np.linalg.norm(self.cluster_centers_ - old_centers, axis=1) < self.tol):
                break

    def predict(self, X):
        distances = np.sqrt(((X[:, np.newaxis] - self.cluster_centers_)**2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        return labels

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

class CustomGaussianMixture:
    def __init__(self, n_components=1, max_iter=100, tol=1e-3, random_state=None, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False

    def fit(self, X):
        n_samples, n_features = X.shape
        kmeans = CustomKMeans(n_clusters=self.n_components, random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        self.means_ = kmeans.cluster_centers_
        self.weights_ = np.array([np.mean(labels == i) for i in range(self.n_components)])
        self.covariances_ = np.array([np.cov(X[labels == i], rowvar=False) + np.eye(n_features) * self.reg_covar for i in range(self.n_components)])
        
        log_likelihood_old = -np.inf
        for i in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            log_likelihood_new = self._compute_log_likelihood(X)
            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                self.converged_ = True
                break
            log_likelihood_old = log_likelihood_new

    def _e_step(self, X):
        log_probs = np.array([self.weights_[j] * multivariate_normal(self.means_[j], self.covariances_[j], allow_singular=True).logpdf(X)
                              for j in range(self.n_components)]).T
        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        log_probs = log_probs - max_log_probs
        responsibilities = np.exp(log_probs) / np.sum(np.exp(log_probs), axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.weights_ = responsibilities.sum(axis=0) / n_samples
        self.means_ = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, np.newaxis]
        for i in range(self.n_components):
            diff = X - self.means_[i]
            self.covariances_[i] = np.dot(responsibilities[:, i] * diff.T, diff) / responsibilities[:, i].sum() + np.eye(n_features) * self.reg_covar

    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for j in range(self.n_components):
            log_likelihood += np.sum(np.log(np.maximum(np.dot(self.weights_[j], multivariate_normal(self.means_[j], self.covariances_[j], allow_singular=True).pdf(X)), 1e-32)))
        return log_likelihood

    def score_samples(self, X):
        log_probs = np.log(np.array([multivariate_normal(self.means_[j], self.covariances_[j], allow_singular=True).pdf(X) * self.weights_[j]
                                     for j in range(self.n_components)]).sum(axis=0))
        return log_probs
    




class SimpleCustomGaussianMixture:
    def __init__(self, n_components=1, max_iter=100, tol=1e-3, random_state=None, reg_covar=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        np.random.seed(random_state)
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False

    def fit(self, X):
        n_samples, n_features = X.shape
        # Ensure data does not contain infinite or NaN values
        X = np.nan_to_num(X)
        
        # Robust initialization of parameters
        indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[indices]
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.covariances_ = np.array([np.eye(n_features) * self.reg_covar for _ in range(self.n_components)])
        
        log_likelihood_old = -np.inf
        for i in range(self.max_iter):
            responsibilities = self._e_step(X)
            self._m_step(X, responsibilities)
            log_likelihood_new = self._compute_log_likelihood(X)
            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                self.converged_ = True
                break
            log_likelihood_old = log_likelihood_new
            # print(f"Iteration {i}, Log Likelihood: {log_likelihood_new}")

    def _e_step(self, X):
        log_probs = np.zeros((X.shape[0], self.n_components))
        for j in range(self.n_components):
            log_probs[:, j] = np.log(self.weights_[j]) + multivariate_normal(self.means_[j], self.covariances_[j], allow_singular=True).logpdf(X)
        max_log_prob = np.max(log_probs, axis=1, keepdims=True)
        stabilized_log_probs = log_probs - max_log_prob
        probs = np.exp(stabilized_log_probs)
        sum_probs = np.sum(probs, axis=1, keepdims=True)
        responsibilities = probs / sum_probs
        return responsibilities

    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        self.weights_ = responsibilities.sum(axis=0) / n_samples
        self.means_ = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, np.newaxis]
        for i in range(self.n_components):
            diff = X - self.means_[i]
            self.covariances_[i] = np.dot(responsibilities[:, i] * diff.T, diff) / responsibilities[:, i].sum() + np.eye(n_features) * self.reg_covar

    def _compute_log_likelihood(self, X):
        log_likelihood = 0
        for j in range(self.n_components):
            dist = multivariate_normal(self.means_[j], self.covariances_[j], allow_singular=True)
            likelihood_contrib = dist.pdf(X) * self.weights_[j]
            log_likelihood += np.sum(np.log(np.maximum(likelihood_contrib, 1e-32)))
        return log_likelihood
    def score_samples(self, X):
        """Calculate the weighted log probabilities for each sample in X under the model."""
        log_probs = np.zeros((X.shape[0], self.n_components))
        for j in range(self.n_components):
            log_probs[:, j] = multivariate_normal(self.means_[j], self.covariances_[j], allow_singular=True).logpdf(X)
        weighted_log_probs = log_probs + np.log(self.weights_)

        max_log_probs = np.max(weighted_log_probs, axis=1, keepdims=True)
        sum_exp_log_probs = np.sum(np.exp(weighted_log_probs - max_log_probs), axis=1, keepdims=True)
        log_sum_exp_log_probs = np.log(sum_exp_log_probs) + max_log_probs
        return log_sum_exp_log_probs.ravel()


preprocessor = DataPreprocessor('../dataset/Train_data.csv')
X_preprocessed = preprocessor.get_preprocessed_data()
y = preprocessor.target.apply(lambda x: 0 if x == 'normal' else 1)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

gmm = SimpleCustomGaussianMixture(n_components=10, random_state=42)
gmm.fit(X_train[y_train == 0])

log_probs = gmm.score_samples(X_test)
threshold = np.percentile(log_probs, 46)
predictions = log_probs < threshold

print(classification_report(y_test, predictions.astype(int)))
print(confusion_matrix(y_test, predictions.astype(int)))
