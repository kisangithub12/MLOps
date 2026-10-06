from pathlib import Path

import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor
from tracking import setup_tracking


def main():
    setup_tracking()

    project_dir = Path(__file__).resolve().parents[1]
    report_dir = project_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

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
        with mlflow.start_run(run_name=f"baseline_{name}"):
            mlflow.set_tags({
                "stage": "baseline",
                "dataset": "UCI Heart Disease",
            })

            mlflow.log_params({
                "model": name,
                "cv_folds": cv.n_splits,
                "cv_shuffle": cv.shuffle,
                "cv_random_state": cv.random_state,
                "training_records": len(X),
                "feature_count": X.shape[1],
                "numeric_imputation": "median",
                "numeric_scaling": "StandardScaler",
                "categorical_imputation": "most_frequent",
                "categorical_encoding": "OneHotEncoder",
            })

            mlflow.log_params({
                f"classifier__{key}": value
                for key, value in model.get_params().items()
            })

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
            fold_results = {"fold": list(range(1, cv.n_splits + 1))}

            for metric in scoring:
                values = scores[f"test_{metric}"]

                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std())
                fold_results[metric] = values

                for fold, value in enumerate(values, start=1):
                    mlflow.log_metric(
                        f"cv_{metric}",
                        float(value),
                        step=fold,
                    )

            mlflow.log_metrics({
                key: value
                for key, value in row.items()
                if key != "model"
            })

            model_slug = name.lower().replace(" ", "_")

            fold_path = report_dir / f"{model_slug}_baseline_folds.csv"
            pd.DataFrame(fold_results).to_csv(
                fold_path,
                index=False,
            )
            mlflow.log_artifact(str(fold_path), artifact_path="reports")

            summary_path = report_dir / f"{model_slug}_baseline_summary.csv"
            pd.DataFrame([row]).to_csv(
                summary_path,
                index=False,
            )
            mlflow.log_artifact(str(summary_path), artifact_path="reports")

            results.append(row)
            print(f"Logged baseline run: {name}")

    summary = pd.DataFrame(results)
    summary.to_csv(
        report_dir / "baseline_cv_results.csv",
        index=False,
    )

    print("\nBaseline cross-validation results:")
    print(summary.round(4).to_string(index=False))
    print("\nSaved to reports/baseline_cv_results.csv")


if __name__ == "__main__":
    main()