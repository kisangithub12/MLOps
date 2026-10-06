from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def main():
    # Download the official Heart Disease dataset.
    dataset = fetch_ucirepo(id=45)

    # Combine input features with the original target.
    data = pd.concat(
        [dataset.data.features, dataset.data.targets],
        axis=1,
    )

    # Resolve the location relative to this script.
    project_dir = Path(__file__).resolve().parents[1]
    output_dir = project_dir / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "heart_disease.csv"
    data.to_csv(output_file, index=False)

    print(f"Saved to: {output_file}")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    print("\nMissing values:")
    print(data.isna().sum())
    print("\nOriginal target counts:")
    print(data["num"].value_counts().sort_index())


if __name__ == "__main__":
    main()