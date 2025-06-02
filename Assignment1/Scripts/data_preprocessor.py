import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def impute_missing_values(data, strategy='mean'):
    """
    Fill missing values in the dataset.
    For numeric columns: use mean, median, or mode.
    For non-numeric (text) columns: always use mode.
    """
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(data[column]):
                if strategy == 'mean':
                    fill_value = data[column].mean()
                elif strategy == 'median':
                    fill_value = data[column].median()
                elif strategy == 'mode':
                    fill_value = data[column].mode()[0]
                else:
                    raise ValueError("Unknown strategy: choose 'mean', 'median', or 'mode'")
            else:
                
                fill_value = data[column].mode()[0]
            data[column].fillna(fill_value, inplace=True)
    return data  #took ChatGPt help in this step as I was hitting erorrs

def remove_duplicates(data):
    """
    Remove duplicate rows from the dataset.
    """
    return data.drop_duplicates()  # drop_duplicates() removes any rows that are the same

def normalize_data(data, method='minmax'):
    """
    Normalize numeric columns.
    method='minmax' scales between 0 and 1.
    method='standard' scales to mean 0, std 1.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data_copy = data.copy()  # Make a copy to keep the original safe
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unknown method: choose 'minmax' or 'standard'")
    data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])
    return data_copy

def remove_redundant_features(data, threshold=0.9):
    """
    Remove features (columns) that are too similar (correlation > threshold).
    Only checks numeric columns.
    """
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr().abs()  # Get absolute correlation values
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return data.drop(columns=to_drop)
# This fills missing values with the average
# to keep as much data as possible.
