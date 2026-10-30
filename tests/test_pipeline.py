import pandas as pd

from src.pipeline import evaluate_model, load_data, preprocess, split, train_model


def test_load_data_has_target():
    df = load_data()
    assert "target" in df.columns
    assert len(df) > 100


def test_preprocess_removes_duplicates_and_nans():
    df = pd.DataFrame({"a": [1, 1, None, 2], "target": [0, 0, 1, 1]})
    clean = preprocess(df)
    assert clean.isna().sum().sum() == 0
    assert len(clean) == 2


def test_model_accuracy_above_threshold():
    df = preprocess(load_data())
    X_train, X_test, y_train, y_test = split(df)
    model = train_model(X_train, y_train)
    assert evaluate_model(model, X_test, y_test) > 0.90
