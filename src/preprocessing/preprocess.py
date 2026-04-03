import pandas as pd
import numpy as np
import os

print("--- Starting preprocessing script ---")

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
    df = df.dropna()
    
    print("Step 1 completed.")
    print(f"Size after cleaning: {df.shape[0]} rows.")

    # ==========================================
    # 2. OUTLIER TREATMENT
    # ==========================================

    print("\nHandling outliers...")
    size_before_outliers = df.shape[0]
    
    col_outlier = 'loan_amt_outstanding'
    
    if col_outlier in df.columns:
        # Context: Extreme loan amounts can severely skew the learning process of our models.
        # We use the IQR method because it is highly robust and 
        # does not assume that our data follows a perfect normal distribution.
        Q1 = df[col_outlier].quantile(0.25)
        Q3 = df[col_outlier].quantile(0.75)
        IQR = Q3 - Q1

        # Filtering out the values that fall outside the acceptable boundaries
        df = df[(df[col_outlier] >= Q1 - 1.5 * IQR) & (df[col_outlier] <= Q3 + 1.5 * IQR)]
        
        outliers_removed = size_before_outliers - df.shape[0]
        print(f"Outliers removed from {col_outlier}: {outliers_removed}")
        print(f"Size after outlier removal: {df.shape[0]} rows.")
    else:
        print(f"Warning: '{col_outlier}' column not found.")

    # ==========================================
    # 3. FEATURE ENGINEERING
    # ==========================================

    print("\nEngineering new features...")

    # Transforming continuous employment years into categorical bins.
    # This non-linear transformation helps predictive models identify distinct 
    # risk profiles (e.g., precarious vs. stable employment) more easily.
    
    if 'years_employed' in df.columns:
        df['employment_group'] = pd.cut(df['years_employed'],
                                       bins=[-1, 2, 5, 10, 50],
                                       labels=['low', 'mid', 'high', 'very_high'])
        print("'employment_group' feature created.")
        
    # Calculating the Debt-to-Income (DTI) ratio.
    # DTI is a vital credit risk indicator. Combining debt and income into a single 
    # feature provides the model with a direct measure of the borrower's repayment capacity.
    if 'loan_amt_outstanding' in df.columns and 'income' in df.columns:
        df['debt_to_income'] = df['loan_amt_outstanding'] / df['income']
        print("'debt_to_income' feature created.")
    else:
        print("Warning: Could not create 'debt_to_income' (missing columns).")

    print(f"Total columns now: {df.shape[1]}")