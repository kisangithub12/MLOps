from pathlib import Path
import hashlib
import json

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

from tracking import setup_tracking


def main():
    setup_tracking()

    project_dir = Path(__file__).resolve().parents[1]
    model_path = project_dir / "models" / "model.joblib"

    model = joblib.load(model_path)
    test_data = pd.read_csv(
        project_dir / "data" / "processed" / "test.csv"
    )

    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    classifier_name = type(model["classifier"]).__name__
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()

    report_dir = project_dir / "reports"
    plot_dir = report_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(
        run_name=f"test_evaluation_{classifier_name}"
    ):
        mlflow.set_tags({
            "stage": "test_evaluation",
            "dataset": "UCI Heart Disease",
            "model_sha256": model_hash,
        })

        mlflow.log_params({
            "classifier": classifier_name,
            "model_file": "models/model.joblib",
            "test_records": len(test_data),
            "feature_count": X_test.shape[1],
            "positive_class": 1,
            "prediction_threshold": 0.5,
        })

        mlflow.log_params({
            f"classifier__{key}": value
            for key, value in model["classifier"].get_params().items()
        })

        # Evaluate the existing fitted pipeline without retraining.
        predictions = model.predict(X_test)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_test)[:, positive_index]

        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            "roc_auc": float(
                roc_auc_score(y_test, probabilities)
            ),
        }

        mlflow.log_metrics({
            f"test_{name}": value
            for name, value in metrics.items()
        })

        metrics_path = report_dir / "test_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)

        confusion_path = plot_dir / "test_confusion_matrix.png"
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=[0, 1],
            display_labels=["Absent", "Present"],
            cmap="Blues",
        )
        plt.title("Selected Model: Held-out Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig(confusion_path, dpi=200)
        plt.close()

        roc_path = plot_dir / "test_roc_curve.png"
        RocCurveDisplay.from_predictions(y_test, probabilities)
        plt.title("Selected Model: Held-out Test ROC Curve")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close()

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="reports",
        )
        mlflow.log_artifact(
            str(confusion_path),
            artifact_path="figures",
        )
        mlflow.log_artifact(
            str(roc_path),
            artifact_path="figures",
        )

        print("Selected classifier:", classifier_name)
        print("Test records:", len(test_data))

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\nMetrics and plots saved in reports/")
        print("Test evaluation logged to MLflow.")


if __name__ == "__main__":
    main()