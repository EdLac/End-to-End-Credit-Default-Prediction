import pandas as pd
import numpy as np
import os

print("--- Starting preprocessing script ---")

# Define data path
data_path = "data/raw/Loan_Data.csv" 

if os.path.exists(data_path):
    print("Data file found successfully !")
    
    # Load data
    df = pd.read_csv(data_path)
    initial_size = df.shape[0]
    print(f"Initial dataset size: {initial_size} rows and {df.shape[1]} columns.\n")
    
    # ==========================================
    # 1. DATA CLEANING & MISSING VALUES
    # ==========================================
    
    # Initial data cleaning ensures consistency and avoids noise in model training
    df = df.drop_duplicates()
    
    # Missing values are handled to prevent bias and data loss
    if 'loan_int_rate' in df.columns:
        df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    
    # Drop remaining NaNs
    df = df.dropna()
    
    final_size = df.shape[0]
    dropped_rows = initial_size - final_size
    
    print("Step 1 completed ! ")
    print(f"Size after cleaning: {final_size} rows.")
    print(f"Total rows dropped (duplicates + NaNs): {dropped_rows} rows.")

else:
    print(f"Error: Cannot find file at {data_path}")