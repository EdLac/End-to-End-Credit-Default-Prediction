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
MODELS_DIR = "models"

def load_data():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").squeeze()

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
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.05, random_state=RANDOM_STATE, eval_metric="logloss"),
    }

def train_and_log_model(model_name, model, X_train, y_train, X_val, y_val):
    # Respect de la consigne 19 : un modele = un experiment
    mlflow.set_experiment(f"Exp_{model_name}")
    
    with mlflow.start_run(run_name=f"Run_{model_name}"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        
        # Log automatique des hyperparametres
        if hasattr(model, 'get_params'):
            mlflow.log_params(model.get_params())

        # Sauvegarde pour maintenance de l'application (app.py / inference.py)
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        columns_path = os.path.join(MODELS_DIR, f"{model_name}_columns.json")
        
        joblib.dump(model, model_path)
        with open(columns_path, "w", encoding="utf-8") as f:
            json.dump(list(X_train.columns), f, ensure_ascii=False, indent=2)

        # Log des fichiers dans MLflow pour la tracabilite (consigne 18)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(columns_path)

        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        else:
            mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Success: {model_name} | AUC: {metrics['roc_auc']:.4f}")
        return metrics["roc_auc"]

def main():
    X_train, y_train, X_val, y_val = load_data()
    models = get_models()

    for model_name, model in models.items():
        train_and_log_model(model_name, model, X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    main()