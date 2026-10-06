import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make modules in scripts/ available to these tests.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from preprocessing import build_preprocessor  # noqa: E402


@pytest.fixture
def training_data():
    return pd.DataFrame({
        "age": [40.0, 50.0, 60.0],
        "trestbps": [120.0, 130.0, 140.0],
        "chol": [200.0, 220.0, 240.0],
        "thalach": [140.0, 150.0, 160.0],
        "oldpeak": [0.0, 1.0, 2.0],
        "sex": [0, 1, 1],
        "cp": [1, 2, 4],
        "fbs": [0, 0, 1],
        "restecg": [0, 1, 2],
        "exang": [0, 0, 1],
        "slope": [1, 2, 3],
        "ca": [0.0, 1.0, 2.0],
        "thal": [3.0, 3.0, 7.0],
    })


def to_dense(values):
    return values.toarray() if hasattr(values, "toarray") else values


def test_missing_values_are_imputed(training_data):
    """Missing numeric and categorical values produce finite outputs."""
    preprocessor = build_preprocessor()
    preprocessor.fit(training_data)

    sample = training_data.iloc[[0]].copy()
    sample.loc[:, "age"] = np.nan
    sample.loc[:, "thal"] = np.nan

    transformed = to_dense(preprocessor.transform(sample))

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed).all()


def test_unknown_category_is_handled(training_data):
    """An unseen category does not fail or change feature dimensions."""
    preprocessor = build_preprocessor()
    preprocessor.fit(training_data)

    known_sample = training_data.iloc[[0]].copy()
    unseen_sample = known_sample.copy()
    unseen_sample.loc[:, "cp"] = 99

    known_output = to_dense(preprocessor.transform(known_sample))
    unseen_output = to_dense(preprocessor.transform(unseen_sample))

    assert unseen_output.shape == known_output.shape
    assert np.isfinite(unseen_output).all()


def test_transform_uses_training_statistics(training_data):
    """Transforming new data must not refit imputation or scaling."""
    preprocessor = build_preprocessor()
    preprocessor.fit(training_data)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]

    original_statistics = imputer.statistics_.copy()
    original_means = scaler.mean_.copy()

    # Training ages are 40, 50, 60: median and mean are both 50.
    assert original_statistics[0] == pytest.approx(50.0)
    assert original_means[0] == pytest.approx(50.0)

    new_data = training_data.iloc[[0]].copy()
    new_data.loc[:, "age"] = 1000.0
    preprocessor.transform(new_data)

    np.testing.assert_array_equal(
        imputer.statistics_,
        original_statistics,
    )
    np.testing.assert_array_equal(
        scaler.mean_,
        original_means,
    )

    # Missing age should use the training median, then scale to zero.
    new_data.loc[:, "age"] = np.nan
    transformed = to_dense(preprocessor.transform(new_data))

    assert transformed[0, 0] == pytest.approx(0.0)