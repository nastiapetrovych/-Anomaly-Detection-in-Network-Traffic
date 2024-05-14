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
from gmm_own import CustomGaussianMixture

preprocessor = DataPreprocessor('../dataset/Train_data.csv')

X_preprocessed = preprocessor.get_preprocessed_data()
y = preprocessor.target.apply(lambda x: 0 if x == 'normal' else 1)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

normal_precisions, normal_recalls, normal_f1s = [], [], []
anomaly_precisions, anomaly_recalls, anomaly_f1s = [], [], []
accuracies = []

n_components = 7
pt = 46

range_limit = 20

for i in range(1, range_limit):
    gmm_normal = CustomGaussianMixture(n_components=i, random_state=42)
    gmm_normal.fit(X_train[y_train == 0])

    log_probs_normal = gmm_normal.score_samples(X_test)


    gmm_anomaly = CustomGaussianMixture(n_components=i, random_state=42)
    gmm_anomaly.fit(X_train[y_train == 1])

    log_probs_anomaly = gmm_anomaly.score_samples(X_test)

    predictions = log_probs_normal < log_probs_anomaly

    report = classification_report(y_test, predictions, output_dict=True)
    normal_precisions.append(report['0']['precision'])
    normal_recalls.append(report['0']['recall'])
    normal_f1s.append(report['0']['f1-score'])

    anomaly_precisions.append(report['1']['precision'])
    anomaly_recalls.append(report['1']['recall'])
    anomaly_f1s.append(report['1']['f1-score'])

    accuracies.append(report['accuracy'])
    print(i)

plt.figure(figsize=(12, 8))

plt.plot(range(1, range_limit), normal_precisions, label='Normal Precision', marker='o', color='blue')
plt.plot(range(1, range_limit), normal_recalls, label='Normal Recall', marker='o', color='cyan')
plt.plot(range(1, range_limit), normal_f1s, label='Normal F1-Score', marker='o', color='purple')

plt.plot(range(1, range_limit), anomaly_precisions, label='Anomaly Precision', marker='x', color='red')
plt.plot(range(1, range_limit), anomaly_recalls, label='Anomaly Recall', marker='x', color='orange')
plt.plot(range(1, range_limit), anomaly_f1s, label='Anomaly F1-Score', marker='x', color='green')

plt.title('GMM Classification Metrics for Normal and Anomaly at Different Percentiles')
plt.xlabel('Threshold')
plt.ylabel('Metric Score')
plt.legend()
plt.grid(True)
plt.show()
