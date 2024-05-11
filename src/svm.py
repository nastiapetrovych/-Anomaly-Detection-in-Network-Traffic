"""
SVM implementation
"""

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


class SVM:
    def __init__(self, features_preprocessed, target):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features_preprocessed, target,
                                                                                test_size=0.2, random_state=42)

    def train(self):
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        custom_svm_models = {}

        # Training SVM with different kernels
        for kernel in kernels:
            svm = SVC(kernel=kernel, C=1.0, random_state=42)
            svm.fit(self.X_train, self.y_train)
            y_pred_custom = svm.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred_custom)
            report = classification_report(self.y_test, y_pred_custom)
            custom_svm_models[kernel] = {'model': svm, 'accuracy': accuracy, 'report': report}

        return{k: v['accuracy'] for k, v in custom_svm_models.items()}
