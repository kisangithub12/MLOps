from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def main():
    project_dir = Path(__file__).resolve().parents[1]
    model_path = project_dir / "models" / "model.joblib"

    test_data = pd.read_csv(
        project_dir / "data" / "processed" / "test.csv"
    )
    X = test_data.drop(columns=["target"])

    model = joblib.load(model_path)

    assert isinstance(model, Pipeline), "Saved model must be a pipeline."
    assert "preprocessing" in model.named_steps
    assert "classifier" in model.named_steps

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    # Verify that loading the same artifact again preserves predictions.
    reloaded_model = joblib.load(model_path)

    np.testing.assert_array_equal(
        predictions,
        reloaded_model.predict(X),
    )
    np.testing.assert_allclose(
        probabilities,
        reloaded_model.predict_proba(X),
    )

    # Verify that fitted imputers handle missing feature values.
    missing_sample = X.iloc[[0]].astype("float64").copy()
    missing_sample.loc[:, "age"] = np.nan
    missing_sample.loc[:, "thal"] = np.nan

    missing_probabilities = model.predict_proba(missing_sample)
    assert np.isfinite(missing_probabilities).all()

    print("Verification passed.")
    print("Saved pipeline includes preprocessing and classifier.")
    print(f"Predictions verified for {len(X)} records.")
    print("Reloaded predictions and probabilities match.")
    print("Missing-value handling works.")


if __name__ == "__main__":
    main()