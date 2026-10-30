"""Shared ML logic."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    data = load_breast_cancer(as_frame=True)
    return data.frame


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().dropna().reset_index(drop=True)


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def train_model(X_train, y_train, n_estimators: int = 100, seed: int = 42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> float:
    return float(accuracy_score(y_test, model.predict(X_test)))


def run_training(output_dir: str = "artifacts") -> float:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = preprocess(load_data())
    X_train, X_test, y_train, y_test = split(df)
    model = train_model(X_train, y_train)
    acc = evaluate_model(model, X_test, y_test)

    joblib.dump(model, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps({"accuracy": acc}, indent=2))
    print(f"Training complete. Accuracy = {acc:.4f}")
    return acc
