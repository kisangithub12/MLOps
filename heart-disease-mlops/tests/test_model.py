import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from preprocessing import build_preprocessor  # noqa: E402


@pytest.fixture
def fitted_model():
    X = pd.DataFrame({
        "age": [40, 45, 50, 55, 60, 65],
        "trestbps": [110, 120, 125, 130, 140, 150],
        "chol": [180, 200, 210, 230, 250, 270],
        "thalach": [170, 165, 155, 145, 130, 120],
        "oldpeak": [0.0, 0.2, 0.5, 1.0, 2.0, 3.0],
        "sex": [0, 1, 0, 1, 0, 1],
        "cp": [1, 2, 2, 3, 4, 4],
        "fbs": [0, 0, 0, 1, 0, 1],
        "restecg": [0, 1, 0, 2, 1, 2],
        "exang": [0, 0, 0, 1, 1, 1],
        "slope": [1, 1, 2, 2, 3, 3],
        "ca": [0.0, 0.0, 1.0, 1.0, 2.0, 3.0],
        "thal": [3.0, 3.0, 3.0, 7.0, 7.0, 7.0],
    })
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = Pipeline([
        ("preprocessing", build_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=10,
            max_depth=3,
            random_state=42,
        )),
    ])
    model.fit(X, y)

    # Separate inference samples, including missing feature values.
    samples = X.iloc[[0, 5]].astype("float64").copy()
    samples.loc[samples.index[0], "age"] = np.nan
    samples.loc[samples.index[1], "thal"] = np.nan

    return model, samples


def test_model_returns_valid_predictions(fitted_model):
    model, samples = fitted_model

    predictions = model.predict(samples)
    probabilities = model.predict_proba(samples)

    assert predictions.shape == (len(samples),)
    assert set(predictions).issubset({0, 1})
    assert probabilities.shape == (len(samples), 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
    )


def test_saved_pipeline_preserves_predictions(fitted_model, tmp_path):
    model, samples = fitted_model
    model_path = tmp_path / "model.joblib"

    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)

    assert isinstance(loaded_model, Pipeline)
    assert "preprocessing" in loaded_model.named_steps
    assert "classifier" in loaded_model.named_steps

    np.testing.assert_array_equal(
        model.predict(samples),
        loaded_model.predict(samples),
    )
    np.testing.assert_allclose(
        model.predict_proba(samples),
        loaded_model.predict_proba(samples),
    )