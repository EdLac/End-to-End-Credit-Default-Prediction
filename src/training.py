import os
import json
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from xgboost import XGBClassifier


RANDOM_STATE = 42
EXPERIMENT_NAME = "credit_default_models"
MODELS_DIR = "models"


def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").squeeze()

    # customer_id is an identifier, not a predictive feature
    if "customer_id" in X_train.columns:
        X_train = X_train.drop(columns=["customer_id"])
    if "customer_id" in X_val.columns:
        X_val = X_val.drop(columns=["customer_id"])

    return X_train, y_train, X_val, y_val


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def get_models():
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            eval_metric="logloss"
        ),
    }
    return models


def get_model_params(model_name, model):
    if model_name == "LogisticRegression":
        return {
            "model_name": model_name,
            "max_iter": model.max_iter,
            "class_weight": str(model.class_weight),
            "random_state": model.random_state,
        }

    if model_name == "RandomForest":
        return {
            "model_name": model_name,
            "n_estimators": model.n_estimators,
            "random_state": model.random_state,
        }

    if model_name == "XGBoost":
        return {
            "model_name": model_name,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "random_state": model.random_state,
            "eval_metric": model.eval_metric,
        }

    return {"model_name": model_name}


def train_and_log_model(model_name, model, X_train, y_train, X_val, y_val):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_proba)
        params = get_model_params(model_name, model)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        columns_path = os.path.join(MODELS_DIR, f"{model_name}_columns.json")
        with open(columns_path, "w", encoding="utf-8") as f:
            json.dump(list(X_train.columns), f, ensure_ascii=False, indent=2)
        mlflow.log_artifact(columns_path)

        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"\n{model_name}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")

        return metrics["roc_auc"], model_path, columns_path


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_val, y_val = load_data()
    models = get_models()

    best_model_name = None
    best_auc = -1
    best_model_path = None
    best_columns_path = None

    for model_name, model in models.items():
        auc, model_path, columns_path = train_and_log_model(
            model_name, model, X_train, y_train, X_val, y_val
        )

        if auc > best_auc:
            best_auc = auc
            best_model_name = model_name
            best_model_path = model_path
            best_columns_path = columns_path

    print("\nBest model selected:")
    print(f"Model: {best_model_name}")
    print(f"Validation ROC AUC: {best_auc:.4f}")
    print(f"Saved model path: {best_model_path}")
    print(f"Saved columns path: {best_columns_path}")


if __name__ == "__main__":
    main()