import json
import joblib
import pandas as pd


MODEL_PATH = "models/LogisticRegression.pkl"
COLUMNS_PATH = "models/LogisticRegression_columns.json"


def load_model_and_columns():
    model = joblib.load(MODEL_PATH)

    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        expected_columns = json.load(f)

    return model, expected_columns


def prepare_input_data(input_data: dict, expected_columns: list[str]) -> pd.DataFrame:
    """
    Convert user input dict into a one-row DataFrame aligned with training columns.
    """

    df = pd.DataFrame([input_data])

    # Remove identifier if ever provided
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # Apply same feature engineering as training
    if "years_employed" in df.columns:
        df["employment_group"] = pd.cut(
            df["years_employed"],
            bins=[-1, 2, 5, 10, 50],
            labels=["low", "mid", "high", "very_high"]
        )

    if "loan_amt_outstanding" in df.columns and "income" in df.columns:
        income_value = df.loc[0, "income"]
        if income_value == 0:
            df["debt_to_income"] = 0
        else:
            df["debt_to_income"] = df["loan_amt_outstanding"] / df["income"]

    # Same encoding logic as training
    df = pd.get_dummies(df, drop_first=True)

    # Add any missing columns expected by the model
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Keep only the columns used during training and in the same order
    df = df[expected_columns]

    return df


def predict_default_risk(input_data: dict) -> dict:
    """
    Returns predicted class and default probability.
    """
    model, expected_columns = load_model_and_columns()
    prepared_df = prepare_input_data(input_data, expected_columns)

    prediction = int(model.predict(prepared_df)[0])
    probability = float(model.predict_proba(prepared_df)[0, 1])

    return {
        "prediction": prediction,
        "default_probability": probability
    }


if __name__ == "__main__":
    sample_input = {
        "credit_lines_outstanding": 1,
        "loan_amt_outstanding": 1000,
        "total_debt_outstanding": 200,
        "income": 145000,
        "years_employed": 6,
        "fico_score": 680
    }

    result = predict_default_risk(sample_input)
    print(result)