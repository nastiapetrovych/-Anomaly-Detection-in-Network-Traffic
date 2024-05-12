import numpy as np
import pandas as pd
from data_preparation import DataPreprocessor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Assuming the DataPreprocessor class is already defined as provided

# Initialize the DataPreprocessor with the path to your CSV file
preprocessor = DataPreprocessor('../dataset/Train_data.csv')

# Get the preprocessed data
X_preprocessed = preprocessor.get_preprocessed_data()
y = preprocessor.target.apply(lambda x: 0 if x == 'normal' else 1)  # Encode labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

n_components = 6
# Fit a Gaussian Mixture Model\
# for i in range(40, 60, 1):
#     gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
#     gmm.fit(X_train[y_train == 0])  # Fit only on normal data
#     print(f"Threshold percentile: {i}")
#     # Predict probabilities and classify as anomaly based on threshold
#     log_probs = gmm.score_samples(X_test)
#     threshold = np.percentile(log_probs, i)  # Adjust threshold as needed based on validation
#     predictions = log_probs < threshold  # Anomalies are where log likelihood is low

#     # Evaluate the model
#     print(classification_report(y_test, predictions))
#     print(confusion_matrix(y_test, predictions))
#     print('\n\n\n')



# gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
# gmm.fit(X_train[y_train == 0])  # Fit only on normal data

# # Predict probabilities and classify as anomaly based on threshold
# log_probs = gmm.score_samples(X_test)
# threshold = np.percentile(log_probs, 10)  # Adjust threshold as needed based on validation
# predictions = log_probs < threshold  # Anomalies are where log likelihood is low

# # Evaluate the model
# print(classification_report(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming `log_probs` and `y_test` are defined as your log-likelihoods from the GMM and the true labels
import matplotlib.pyplot as plt

# Fit GMM and get log-likelihood scores
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
gmm.fit(X_train[y_train == 0])
log_probs = gmm.score_samples(X_test)

# Plot histogram of log-likelihood scores
plt.hist(log_probs, bins=100, alpha=0.75, color='blue', label='Log-Likelihood Scores')
plt.axvline(x=np.percentile(log_probs, 50), color='r', linestyle='dashed', linewidth=1, label='50th Percentile')
plt.title('Histogram of Log-Likelihood Scores with 50th Percentile')
plt.xlabel('Log-Likelihood Score')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Print actual percentile value
print("50th Percentile Score:", np.percentile(log_probs, 50))

