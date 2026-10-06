from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor
from tracking import setup_tracking


def main():
    setup_tracking()

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
            LogisticRegression(
                max_iter=2000,
                random_state=42,
            ),
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
    winner_metrics = None
    winner_candidate_run_id = None

    for name, (classifier, parameter_grid) in experiments.items():
        with mlflow.start_run(run_name=f"tuning_{name}"):
            mlflow.set_tags({
                "stage": "hyperparameter_tuning",
                "dataset": "UCI Heart Disease",
            })

            mlflow.log_params({
                "model": name,
                "selection_metric": "roc_auc",
                "cv_folds": cv.n_splits,
                "cv_shuffle": cv.shuffle,
                "random_state": 42,
                "training_records": len(X),
                "feature_count": X.shape[1],
                "parameter_grid": str(parameter_grid),
                "numeric_imputation": "median",
                "numeric_scaling": "StandardScaler",
                "categorical_imputation": "most_frequent",
                "categorical_encoding": "OneHotEncoder",
            })

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

            cv_results = search.cv_results_

            tuning_path = report_dir / f"{name}_tuning.csv"
            pd.DataFrame(cv_results).to_csv(
                tuning_path,
                index=False,
            )
            mlflow.log_artifact(
                str(tuning_path),
                artifact_path="reports",
            )

            best_candidate_run_id = None

            # Record every candidate using the completed search results.
            for index, candidate_params in enumerate(
                cv_results["params"]
            ):
                with mlflow.start_run(
                    run_name=f"{name}_candidate_{index + 1:02d}",
                    nested=True,
                ) as candidate_run:
                    mlflow.set_tags({
                        "stage": "tuning_candidate",
                        "model": name,
                        "best_for_model": str(
                            index == search.best_index_
                        ).lower(),
                    })

                    # Include both fixed and tuned classifier settings.
                    full_params = {
                        f"classifier__{key}": value
                        for key, value in classifier.get_params().items()
                    }
                    full_params.update(candidate_params)

                    mlflow.log_params(full_params)
                    mlflow.log_params({
                        "cv_folds": cv.n_splits,
                        "random_state": 42,
                        "training_records": len(X),
                    })

                    candidate_metrics = {}

                    for metric in scoring:
                        candidate_metrics[f"{metric}_mean"] = float(
                            cv_results[f"mean_test_{metric}"][index]
                        )
                        candidate_metrics[f"{metric}_std"] = float(
                            cv_results[f"std_test_{metric}"][index]
                        )

                        for fold in range(cv.n_splits):
                            mlflow.log_metric(
                                f"cv_{metric}",
                                float(
                                    cv_results[
                                        f"split{fold}_test_{metric}"
                                    ][index]
                                ),
                                step=fold + 1,
                            )

                    candidate_metrics["rank_roc_auc"] = int(
                        cv_results["rank_test_roc_auc"][index]
                    )
                    mlflow.log_metrics(candidate_metrics)

                    if index == search.best_index_:
                        best_candidate_run_id = (
                            candidate_run.info.run_id
                        )

            row = {
                "model": name,
                "best_parameters": str(search.best_params_),
            }
            best_metrics = {}

            for metric in scoring:
                for statistic in ("mean", "std"):
                    key = f"{metric}_{statistic}"
                    value = float(
                        cv_results[
                            f"{statistic}_test_{metric}"
                        ][search.best_index_]
                    )
                    row[key] = value
                    best_metrics[key] = value

            summaries.append(row)

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(best_metrics)
            mlflow.set_tag(
                "best_candidate_run_id",
                best_candidate_run_id,
            )

            print(f"\n{name}")
            print(f"Best parameters: {search.best_params_}")
            print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
            print(f"Logged {len(cv_results['params'])} candidates")

            if search.best_score_ > winner_score:
                winner_name = name
                winner_score = search.best_score_
                winner_pipeline = search.best_estimator_
                winner_metrics = best_metrics
                winner_candidate_run_id = best_candidate_run_id

    summary = pd.DataFrame(summaries)
    summary_path = report_dir / "tuned_cv_results.csv"
    summary.to_csv(summary_path, index=False)

    # Includes preprocessing and classifier fitted on training data.
    model_path = model_dir / "model.joblib"
    joblib.dump(winner_pipeline, model_path)

    with mlflow.start_run(run_name=f"selected_{winner_name}"):
        mlflow.set_tags({
            "stage": "selected_model",
            "dataset": "UCI Heart Disease",
            "source_candidate_run_id": winner_candidate_run_id,
        })

        mlflow.log_params({
            "model": winner_name,
            "selection_metric": "roc_auc",
            "training_records": len(X),
            "feature_count": X.shape[1],
            "cv_folds": cv.n_splits,
            "random_state": 42,
        })

        mlflow.log_params({
            f"classifier__{key}": value
            for key, value in winner_pipeline.named_steps[
                "classifier"
            ].get_params().items()
        })
        mlflow.log_metrics(winner_metrics)

        mlflow.log_artifact(
            str(summary_path),
            artifact_path="reports",
        )
        mlflow.log_artifact(
            str(model_path),
            artifact_path="model_files",
        )

        input_example = X.head(5).astype("float64")

        # Explicit format avoids the skops untrusted-types error.
        model_info = mlflow.sklearn.log_model(
            sk_model=winner_pipeline,
            name="heart_disease_pipeline",
            input_example=input_example,
            serialization_format="cloudpickle",
        )

        print(f"\nMLflow model URI: {model_info.model_uri}")

    print("\nTuned comparison:")
    print(summary.round(4).to_string(index=False))
    print(f"\nSelected model: {winner_name}")
    print("Saved fitted pipeline to models/model.joblib")


if __name__ == "__main__":
    main()