from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt


project_dir = Path(__file__).resolve().parents[1]

model = joblib.load(project_dir / "models" / "model.joblib")
test_data = pd.read_csv(
    project_dir / "data" / "processed" / "test.csv"
)

X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

predictions = model.predict(X_test)

# Find the probability column for disease-present class 1.
positive_index = list(model.classes_).index(1)
probabilities = model.predict_proba(X_test)[:, positive_index]

metrics = {
    "accuracy": accuracy_score(y_test, predictions),
    "precision": precision_score(y_test, predictions, zero_division=0),
    "recall": recall_score(y_test, predictions, zero_division=0),
    "roc_auc": roc_auc_score(y_test, probabilities),
}

report_dir = project_dir / "reports"
plot_dir = report_dir / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)

with open(report_dir / "test_metrics.json", "w") as file:
    json.dump(metrics, file, indent=2)

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Absent", "Present"],
    cmap="Blues",
)
plt.title("Selected Model: Held-out Test Confusion Matrix")
plt.tight_layout()
plt.savefig(plot_dir / "test_confusion_matrix.png", dpi=200)
plt.close()

RocCurveDisplay.from_predictions(y_test, probabilities)
plt.title("Selected Model: Held-out Test ROC Curve")
plt.tight_layout()
plt.savefig(plot_dir / "test_roc_curve.png", dpi=200)
plt.close()

print("Selected classifier:", type(model["classifier"]).__name__)
print("Test records:", len(test_data))

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nMetrics and plots saved in reports/")