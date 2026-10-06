from pathlib import Path

import mlflow

PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "heart-disease-classification"


def setup_tracking():
    database_path = PROJECT_DIR / "mlflow.db"
    artifact_dir = PROJECT_DIR / "mlartifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{database_path}")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=artifact_dir.as_uri(),
        )

    mlflow.set_experiment(EXPERIMENT_NAME)


if __name__ == "__main__":
    setup_tracking()
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")