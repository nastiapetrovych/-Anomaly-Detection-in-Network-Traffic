import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from data_preparation import DataPreprocessor
from gmm_own import CustomKMeans

preprocessor = DataPreprocessor('../dataset/Train_data.csv')
X_preprocessed = preprocessor.get_preprocessed_data()
y = preprocessor.target.apply(lambda x: 0 if x == 'normal' else 1)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, random_state=42)

n_clusters = 9
custom_kmeans = CustomKMeans(n_clusters=n_clusters, random_state=42)
custom_kmeans.fit(X_train)

cluster_labels = np.zeros(n_clusters)
for i in range(n_clusters):
    labels, counts = np.unique(y_train[custom_kmeans.predict(X_train) == i], return_counts=True)
    cluster_labels[i] = labels[np.argmax(counts)]

test_labels = custom_kmeans.predict(X_test)
predictions = np.array([cluster_labels[label] for label in test_labels])

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
