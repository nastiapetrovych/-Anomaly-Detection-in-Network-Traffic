import matplotlib.pyplot as plt
import pandas as pd
from data_preparation import DataPreprocessor

preprocessor = DataPreprocessor('../dataset/Train_data.csv')
df = preprocessor.data
y = df['class'].apply(lambda x: 0 if x == 'normal' else 1)

# Define number of plots per figure
plots_per_figure = 20  # Set how many plots you want per figure
num_features = len(df.columns) - 1

# Iterate over features and create new figures as necessary
for i, column in enumerate(df.drop(columns=['class']).columns):
    if i % plots_per_figure == 0:
        plt.figure(figsize=(15, 10))
    ax = plt.subplot(5, 4, (i % plots_per_figure) + 1)
    plt.hist(df.loc[y == 0, column], bins=30, alpha=0.5, color='green', label='Normal')
    plt.hist(df.loc[y == 1, column], bins=30, alpha=0.5, color='red', label='Anomaly')
    ax.set_title(column)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    if (i + 1) % plots_per_figure == 0 or (i + 1) == num_features:
        plt.show()
