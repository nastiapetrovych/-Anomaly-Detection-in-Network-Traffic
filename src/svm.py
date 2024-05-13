"""
SVM implementation
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


class SVM:
    def __init__(self, data, target, features_result_df, preprocessed_data):
        self.data = data
        self.target = target
        self.features_result_df = features_result_df
        self.features = data.drop(columns=[target])
        self.label_encoder = LabelEncoder()
        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid']

        self.features_preprocessed = preprocessed_data

    def train_and_measure_accuracy(self):
        custom_svm_models = {}
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.data[self.target],
                                                                           test_size=0.2, random_state=42)
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        # Training SVM with different kernels
        for kernel in self.kernels:
            svm = SVC(kernel=kernel, C=1.0, random_state=42)
            svm.fit(X_train, y_train_encoded)
            y_pred_custom = svm.predict(X_test)
            accuracy = accuracy_score(y_test_encoded, y_pred_custom)
            report = classification_report(y_test_encoded, y_pred_custom, target_names=self.label_encoder.classes_)
            custom_svm_models[kernel] = {'model': svm, 'accuracy': accuracy, 'report': report}

        return {k: v['accuracy'] for k, v in custom_svm_models.items()}

    def predict(self, X_test):
        X_train = self.features_preprocessed['train']
        y_train_encoded = self.label_encoder.fit_transform(self.target)
        for kernel in self.kernels:
            model = SVC(kernel=kernel, C=1.0, random_state=42)
            model.fit(X_train, y_train_encoded)

            y_pred_custom = model.predict(X_test)
            X_test_pca = PCA(n_components=2).fit_transform(X_test)
            self.visualize_results(X_test_pca, y_pred_custom, kernel)

    def visualize_results(self, X_test_pca, y_pred_custom, kernel):
        plt.figure(figsize=(10, 6))
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred_custom, cmap='coolwarm', s=20)
        plt.title(f'SVM Predictions with {kernel} kernel (PCA)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Prediction')
        plt.savefig(f'result_{kernel}.png')
        plt.show()
