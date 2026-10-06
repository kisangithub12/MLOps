from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

project_dir = Path(__file__).resolve().parents[1]
data = pd.read_csv(project_dir / "data" / "raw" / "heart_disease.csv")

# Inputs exclude the original target.
X = data.drop(columns=["num"])
y = (data["num"] > 0).astype(int).rename("target")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

output_dir = project_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

train_data = X_train.copy()
train_data["target"] = y_train

test_data = X_test.copy()
test_data["target"] = y_test

train_data.to_csv(output_dir / "train.csv", index=False)
test_data.to_csv(output_dir / "test.csv", index=False)

print(f"Training rows: {len(train_data)}")
print(f"Test rows: {len(test_data)}")
print("\nTraining class counts:")
print(y_train.value_counts().sort_index())
print("\nTest class counts:")
print(y_test.value_counts().sort_index())
print(f"\nSaved to: {output_dir}")