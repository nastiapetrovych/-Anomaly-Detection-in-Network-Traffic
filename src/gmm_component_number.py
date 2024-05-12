import numpy as np
import pandas as pd
from data_preparation import DataPreprocessor
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

preprocessor = DataPreprocessor('../dataset/Train_data.csv')

X_preprocessed = preprocessor.get_preprocessed_data()
y = preprocessor.target.apply(lambda x: 0 if x == 'normal' else 1)  # Encode labels

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

bic_scores = []
for i in range(1, 11):
    gmm = GaussianMixture(n_components=i, covariance_type='full', random_state=42)
    gmm.fit(X_train[y_train == 0])  # Fit only on normal data
    print(f"Number of components: {i}, BIC: {gmm.bic(X_train[y_train == 0])}")
    bic_scores.append(gmm.bic(X_train[y_train == 0]))
    # Predict probabilities and classify as anomaly based on threshold
    log_probs = gmm.score_samples(X_test)
    threshold = np.percentile(log_probs, 25)  # Adjust threshold as needed based on validation
    predictions = log_probs < threshold  # Anomalies are where log likelihood is low

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print('\n\n\n')
print("Minimum BIC score index: ", np.argmin(bic_scores)+1)