import numpy as np
import pandas as pd
from data_preparation import DataPreprocessor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.mixture import GaussianMixture
import numpy as np

# Assuming the DataPreprocessor class is already defined as provided

# Initialize the DataPreprocessor with the path to your CSV file
preprocessor = DataPreprocessor('../dataset/Train_data.csv')

# Get the preprocessed data
X_preprocessed = preprocessor.get_preprocessed_data()
y = preprocessor.target.apply(lambda x: 0 if x == 'normal' else 1)  # Encode labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

n_components = 6

# Initialize lists to store metrics for each class
normal_precisions, normal_recalls, normal_f1s = [], [], []
anomaly_precisions, anomaly_recalls, anomaly_f1s = [], [], []
accuracies = []

# Define n_components based on your model's configuration
n_components = 6  # Adjust this as per your GMM setup

for i in range(1, 60):
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm.fit(X_train[y_train == 0])  # Fit only on normal data

    # Predict probabilities and classify based on threshold
    log_probs = gmm.score_samples(X_test)
    threshold = np.percentile(log_probs, i)
    predictions = log_probs < threshold

    # Get precision, recall, and F1-score for each class
    report = classification_report(y_test, predictions, output_dict=True)
    normal_precisions.append(report['0']['precision'])
    normal_recalls.append(report['0']['recall'])
    normal_f1s.append(report['0']['f1-score'])

    anomaly_precisions.append(report['1']['precision'])
    anomaly_recalls.append(report['1']['recall'])
    anomaly_f1s.append(report['1']['f1-score'])

    accuracies.append(report['accuracy'])

# Plotting the metrics
plt.figure(figsize=(12, 8))

# Normal class metrics
plt.plot(range(1, 60), normal_precisions, label='Normal Precision', marker='o', color='blue')
plt.plot(range(1, 60), normal_recalls, label='Normal Recall', marker='o', color='cyan')
plt.plot(range(1, 60), normal_f1s, label='Normal F1-Score', marker='o', color='purple')

# Anomaly class metrics
plt.plot(range(1, 60), anomaly_precisions, label='Anomaly Precision', marker='x', color='red')
plt.plot(range(1, 60), anomaly_recalls, label='Anomaly Recall', marker='x', color='orange')
plt.plot(range(1, 60), anomaly_f1s, label='Anomaly F1-Score', marker='x', color='green')

plt.title('GMM Classification Metrics for Normal and Anomaly at Different Percentiles')
plt.xlabel('Percentile Threshold')
plt.ylabel('Metric Score')
plt.legend()
plt.grid(True)
plt.show()
