from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor


def main():
    project_dir = Path(__file__).resolve().parents[1]
    data = pd.read_csv(
        project_dir / "data" / "processed" / "train.csv"
    )

    X = data.drop(columns=["target"])
    y = data["target"]

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

    experiments = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {
                "classifier__C": [0.01, 0.1, 1, 10],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=1,
            ),
            {
                "classifier__max_depth": [None, 5, 10],
                "classifier__min_samples_leaf": [1, 3, 5],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
    }

    report_dir = project_dir / "reports"
    model_dir = project_dir / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    winner_name = None
    winner_score = float("-inf")
    winner_pipeline = None

    for name, (classifier, parameter_grid) in experiments.items():
        pipeline = Pipeline([
            ("preprocessing", build_preprocessor()),
            ("classifier", classifier),
        ])

        search = GridSearchCV(
            pipeline,
            param_grid=parameter_grid,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X, y)

        # Save every parameter combination and its validation scores.
        pd.DataFrame(search.cv_results_).to_csv(
            report_dir / f"{name}_tuning.csv",
            index=False,
        )

        row = {
            "model": name,
            "best_parameters": str(search.best_params_),
        }

        for metric in scoring:
            row[f"{metric}_mean"] = search.cv_results_[
                f"mean_test_{metric}"
            ][search.best_index_]

        summaries.append(row)

        print(f"\n{name}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

        if search.best_score_ > winner_score:
            winner_name = name
            winner_score = search.best_score_
            winner_pipeline = search.best_estimator_

    summary = pd.DataFrame(summaries)
    summary.to_csv(
        report_dir / "tuned_cv_results.csv",
        index=False,
    )

    # Includes fitted preprocessing AND classifier.
    joblib.dump(winner_pipeline, model_dir / "model.joblib")

    print("\nTuned comparison:")
    print(summary.to_string(index=False))
    print(f"\nSelected model: {winner_name}")
    print("Saved fitted pipeline to models/model.joblib")


if __name__ == "__main__":
    main()