import pandas as pd
import numpy as np

print("--- Starting preprocessing script ---")

# Data loading
data_path = "data/raw/Loan_Data.csv" 
df = pd.read_csv(data_path)

# Record initial size for Before/After comparison
initial_size = df.shape[0]
print(f"Initial dataset size: {initial_size} rows.")

# ==========================================
# 1. DATA CLEANING
# ==========================================
# Initial data cleaning ensures consistency and avoids noise in model training
df = df.drop_duplicates()

# ==========================================
# 2. MISSING VALUES HANDLING
# ==========================================
# Missing values are handled to prevent bias and data loss

# Median imputation for interest rate
if 'loan_int_rate' in df.columns:
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

# Drop remaining NaNs
df = df.dropna()

# Calculate final size and difference
final_size = df.shape[0]
dropped_rows = initial_size - final_size


print(f"Size after cleaning: {final_size} rows.")
print(f"Total rows dropped (duplicates + NaNs): {dropped_rows} rows.")