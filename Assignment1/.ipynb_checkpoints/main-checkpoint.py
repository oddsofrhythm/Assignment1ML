import pandas as pd
from Scripts.data_preprocessor import (
    impute_missing_values,
    remove_duplicates,
    normalize_data,
    remove_redundant_features
)

# 1️⃣ Load the messy data
data = pd.read_csv('Data/messy_data.csv')

# 2️⃣ Impute missing values
imputed_data = impute_missing_values(data, strategy='mean')

# 3️⃣ Remove duplicate rows
deduped_data = remove_duplicates(imputed_data)

# 4️⃣ Normalize numeric data
normalized_data = normalize_data(deduped_data, method='minmax')

# 5️⃣ Remove redundant features (highly correlated)
final_data = remove_redundant_features(normalized_data, threshold=0.9)

# 6️⃣ Save the final clean data
final_data.to_csv('Data/cleaned_data.csv', index=False)

print("✅ Data cleaning complete! Check Data/cleaned_data.csv.")
