from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

project_dir = Path(__file__).resolve().parents[1]
data = pd.read_csv(project_dir / "data" / "raw" / "heart_disease.csv")

print("First five rows:")
print(data.head())

print("\nColumn types:")
print(data.dtypes)

print("\nMissing values:")
print(data.isna().sum())

print("\nDuplicate rows:")
print(data.duplicated().sum())

print("\nNumerical summary:")
print(data.describe())

# Derive the binary target for inspection; leave the raw CSV unchanged.
target = (data["num"] > 0).astype(int)

print("\nClass counts: 0 = absent, 1 = present")
print(target.value_counts().sort_index())

##class-balance plot.

plot_dir = project_dir / "reports" / "figures"
plot_dir.mkdir(parents=True, exist_ok=True)

counts = target.value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    ["Absent (0)", "Present (1)"],
    counts.values,
    color=["steelblue", "coral"],
)
ax.bar_label(bars, padding=3)
ax.set_title("Heart Disease Class Balance")
ax.set_ylabel("Number of patients")
ax.set_ylim(0, counts.max() * 1.15)

fig.tight_layout()
output_file = plot_dir / "class_balance.png"
fig.savefig(output_file, dpi=200)
plt.close(fig)

print(f"\nPlot saved to: {output_file}")

## Histograms for the continuous features and a correlation heatmap


# Continuous measurements—not category codes.
continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

data[continuous_features].hist(
    bins=15,
    figsize=(12, 8),
    color="steelblue",
    edgecolor="white",
)
plt.suptitle("Distributions of Continuous Features")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(plot_dir / "feature_histograms.png", dpi=200)
plt.close()

# Use the binary target; exclude the original 0–4 target.
correlation_data = data[continuous_features].copy()
correlation_data["target"] = target

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    correlation_data.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    ax=ax,
)
ax.set_title("Continuous Features and Binary Target: Pearson Correlation")
fig.tight_layout()
fig.savefig(plot_dir / "correlation_heatmap.png", dpi=200)
plt.close(fig)

print("Histograms and correlation heatmap saved.")