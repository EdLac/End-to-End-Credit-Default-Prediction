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
    df = df.drop_duplicates(subset=["customer_id"])
    
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

    # ==========================================
    # 4. ENCODING CATEGORICAL VARIABLES
    # ==========================================
    print("\nEncoding categorical variables...")
    # Machine learning models require numerical input. We use one-hot encoding
    # to transform categorical text data into binary (0/1) columns.
    df_encoded = pd.get_dummies(df, drop_first=True)
    print(f"Encoding completed. Dataset now has {df_encoded.shape[1]} numeric columns.")

    # ==========================================
    # 5. DEFINE X / y
    # ==========================================
    print("\nDefining features (X) and target (y)...")
    target_col = 'default'
    
    if target_col in df_encoded.columns:
        X = df_encoded.drop(target_col, axis=1)
        y = df_encoded[target_col]
        print(f"Features (X) and Target (y) defined. Total features: {X.shape[1]}")

        # ==========================================
        # 6. TRAIN / VALIDATION / TEST SPLIT
        # ==========================================
        print("\nSplitting data into Train, Validation and Test sets...")
        from sklearn.model_selection import train_test_split
        
        # 60% Train, 20% Val, 20% Test
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)
        
        print(f"Initial Train size: {X_train.shape[0]}")
        print(f"Validation size: {X_val.shape[0]}")
        print(f"Test size: {X_test.shape[0]}")

        # ==========================================
        # 7. CLASS IMBALANCE HANDLING (Only on Train)
        # ==========================================
        print("\nHandling class imbalance on training set...")
        # Context: Class imbalance handling is critical for realistic model performance.
        # We only resample the training set to prevent data leakage.
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print("SMOTE applied successfully on training data!")
        except ImportError:
            print("Warning: SMOTE not available. Falling back to sklearn resample.")
            from sklearn.utils import resample
            train_data = pd.concat([X_train, y_train], axis=1)
            majority = train_data[train_data[target_col] == 0]
            minority = train_data[train_data[target_col] == 1]
            minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
            train_data_upsampled = pd.concat([majority, minority_upsampled])
            X_train = train_data_upsampled.drop(target_col, axis=1)
            y_train = train_data_upsampled[target_col]
            print("Sklearn resample applied successfully.")

        # ==========================================
        # 8. EXPORTING PROCESSED DATA
        # ==========================================
        print("\nSaving processed files to disk...")
        os.makedirs("data/processed", exist_ok=True)
        
        datasets = {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test
        }
        
        for name, data in datasets.items():
            data.to_csv(f"data/processed/{name}.csv", index=False)
        
        print("All 6 files (Train, Val, Test) saved successfully in 'data/processed/'!")
    else:
        print(f"Error: Target column '{target_col}' not found.")

else:
    print(f"Error: Cannot find file at {data_path}")

print("\n--- Preprocessing script finished ---")