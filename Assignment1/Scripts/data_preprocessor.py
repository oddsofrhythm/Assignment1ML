def impute_missing_values(data, strategy='mean'):
    for column in data.columns:
        if data[column].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(data[column]):
                if strategy == 'mean':
                    fill_value = data[column].mean()
                elif strategy == 'median':
                    fill_value = data[column].median()
                elif strategy == 'mode':
                    fill_value = data[column].mode()[0]
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
            else:
                fill_value = data[column].mode()[0]
            data[column].fillna(fill_value, inplace=True)
    return data  # make sure this line is present

def remove_duplicates(data):
    return data.drop_duplicates()  # make sure this line is present

def normalize_data(data, method='minmax'):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data_copy = data.copy()
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    data_copy[numeric_cols] = scaler.fit_transform(data_copy[numeric_cols])
    return data_copy  # make sure this line is present

def remove_redundant_features(data, threshold=0.9):
    """
    Remove highly correlated numeric columns from the dataset.
    Keeps the non-numeric columns intact.
    """
    # Separate numeric data
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Compute correlation matrix on numeric data
    corr_matrix = numeric_data.corr().abs()
    
    # Identify redundant features
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    
    # Drop redundant numeric columns from the **original dataset** (including non-numeric columns!)
    return data.drop(columns=to_drop)


