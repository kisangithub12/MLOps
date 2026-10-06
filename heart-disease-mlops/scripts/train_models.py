from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


def main():
    project_dir = Path(__file__).resolve().parents[1]
    train_data = pd.read_csv(
        project_dir / "data" / "processed" / "train.csv"
    )

    X = train_data.drop(columns=["target"])
    y = train_data["target"]

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=1,
        ),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc",
    }

    results = []

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessing", build_preprocessor()),
            ("classifier", model),
        ])

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise",
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        results.append(row)

    summary = pd.DataFrame(results)

    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        report_dir / "baseline_cv_results.csv",
        index=False,
    )

    print("\nBaseline cross-validation results:")
    print(summary.round(4).to_string(index=False))
    print("\nSaved to reports/baseline_cv_results.csv")


if __name__ == "__main__":
    main()