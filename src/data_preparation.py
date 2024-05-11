"""
Preprocess data before training
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    def __init__(self, file_path: str):
        self.data = pd.read_csv(file_path)
        self.features = self.data.drop(columns=['class'])
        self.target = self.data['class']
        self.features_result_df = self.get_important_features()

    def get_important_features(self):
        # Label Encoding for categorical features
        label_encoders = {}
        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                le = LabelEncoder()
                self.data[column] = le.fit_transform(self.data[column])
                label_encoders[column] = le

        # Calculate mutual information
        mi_scores = mutual_info_classif(self.features, self.target, discrete_features='auto')

        # Calculate target entropy
        target_entropy = self.calculate_entropy(self.target)

        # Calculate information gain
        information_gain = target_entropy - (target_entropy - mi_scores)

        # Calculate intrinsic values for each feature
        intrinsic_values = np.array([self.intrinsic_value(self.features[col]) for col in self.features.columns])

        # Calculate information gain ratio
        information_gain_ratio = information_gain / intrinsic_values
        ig_df = pd.DataFrame({
            'Feature': self.features.columns,
            'Mutual Information': mi_scores,
            'Intrinsic Value': intrinsic_values,
            'Information Gain Ratio': information_gain_ratio
        })

        ig_df.sort_values(by='Information Gain Ratio', ascending=False)
        return ig_df

    def calculate_entropy(self, feature):
        _, counts = np.unique(feature, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy

    def intrinsic_value(self, feature):
        _, counts = np.unique(feature, return_counts=True)
        probabilities = counts / counts.sum()
        iv = -np.sum(probabilities * np.log2(probabilities))
        return iv

    def visualize_results(self, file_path: str):
        plt.figure(figsize=(14, 7))
        # Plot top features based on mutual information
        ig_df_sorted = self.features_result_df.sort_values(by='Mutual Information', ascending=False)
        plt.subplot(1, 2, 1)
        sns.barplot(x='Mutual Information', y='Feature', data=ig_df_sorted.head(10), palette='viridis')
        plt.title('Top 10 Features by Mutual Information')

        # Plot tip features based on the information gain ratio
        ig_df_sorted = self.features_result_df.sort_values(by='Information Gain Ratio', ascending=False)
        plt.subplot(1, 2, 2)
        sns.barplot(x='Information Gain Ratio', y='Feature', data=ig_df_sorted.head(10), palette='viridis')
        plt.title('Top 10 Features by Information Gain Ratio')

        plt.tight_layout()
        plt.savefig(f'{file_path}/features_result.jpg')
        plt.show()

    def get_preprocessed_data(self):
        categorical_features = [col for col in self.data.columns if self.data[col].dtype == 'object']
        important_categorical_features = \
            self.features_result_df[self.features_result_df['Feature'].isin(categorical_features) &
                                    (self.features_result_df['Information Gain Ratio'] > 0.05)]['Feature'].tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(), important_categorical_features),
                ('num', MinMaxScaler(feature_range=(-1, 1)), self.features.select_dtypes(include=[np.number]).columns)
            ],
            remainder='passthrough'
        )

        features_preprocessed = preprocessor.fit_transform(self.features)
        return features_preprocessed
