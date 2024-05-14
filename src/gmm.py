import numpy as np
import pandas as pd
from data_preparation import DataPreprocessor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

preprocessor = DataPreprocessor('../dataset/Train_data.csv')

X_preprocessed = preprocessor.get_preprocessed_data()
y = preprocessor.target.apply(lambda x: 0 if x == 'normal' else 1)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

n_components = 7
# A sigt bias to type 1 error for better result for anomaly detection
percentile_threshold = 46

gmm = GaussianMixture(n_components=10, covariance_type='full', random_state=42)
gmm.fit(X_train[y_train == 0]) 

log_probs = gmm.score_samples(X_test)
threshold = np.percentile(log_probs, percentile_threshold)
predictions = log_probs < threshold

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))


