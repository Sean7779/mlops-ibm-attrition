import numpy as np

from sklearn.metrics import accuracy_score

from src.train import train_model


def test_model_produces_predictions_with_correct_shape():
    model, metrics, X_test, y_test = train_model(
        "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    predictions = model.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == y_test.shape[0]
    assert predictions.ndim == 1


def test_model_meets_minimum_accuracy_threshold():
    model, metrics, X_test, y_test = train_model(
        "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    )

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.60

    