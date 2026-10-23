from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "DATA SET"
if not DATA_DIR.is_dir():
    DATA_DIR = BASE_DIR

DATASETS = {
    "Iris": DATA_DIR / "Iris.csv",
    "Mobile Legends Skins": DATA_DIR / "mobile_legends_skins_dataset_1500.csv",
}
BINARY_CATEGORICAL_COLUMNS = {"is_limited", "is_event", "is_collab"}


def load_dataset(name: str, path: Path) -> pd.DataFrame:
    """Load a local CSV, with sklearn's Iris data as a fallback."""
    if path.exists():
        data = pd.read_csv(path, skipinitialspace=True)
    elif name == "Iris":
        iris = load_iris(as_frame=True)
        data = iris.frame.rename(
            columns={
                "sepal length (cm)": "SepalLengthCm",
                "sepal width (cm)": "SepalWidthCm",
                "petal length (cm)": "PetalLengthCm",
                "petal width (cm)": "PetalWidthCm",
                "target": "Species",
            }
        )
        data["Species"] = data["Species"].map(
            dict(enumerate(iris.target_names))
        )
        data.insert(0, "Id", np.arange(1, len(data) + 1))
    else:
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    data.columns = data.columns.str.strip()
    for column in data.select_dtypes(include=["object", "str"]).columns:
        data[column] = data[column].str.strip()
    return data


def analyze_dataset(name: str, data: pd.DataFrame) -> None:
    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column
        for column in data.columns
        if column not in numeric_columns
        or column in BINARY_CATEGORICAL_COLUMNS
    ]
    numeric_columns = [
        column
        for column in numeric_columns
        if column not in BINARY_CATEGORICAL_COLUMNS
    ]

    duplicate_rows = data[data.duplicated(keep=False)].sort_values(
        by=data.columns.tolist(), na_position="last"
    )
    null_counts = data.isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    null_rows = data[data.isna().any(axis=1)].sort_values(
        by=data.columns[0], na_position="last"
    )

    zero_counts = pd.Series(
        {
            column: int(np.count_nonzero(data[column].eq(0).fillna(False)))
            for column in data.select_dtypes(include=np.number).columns
        },
        dtype="int64",
    ).sort_values(ascending=False)
    zero_rows = (
        data[data[zero_counts.index].eq(0).any(axis=1)].sort_values(
            by=data.columns[0], na_position="last"
        )
        if not zero_counts.empty
        else data.iloc[0:0]
    )

    print(f"\n{'=' * 72}\nDATASET: {name}")
    print(f"Ukuran data: {data.shape[0]} baris x {data.shape[1]} kolom")
    print(
        f"Kolom kategorikal ({len(categorical_columns)}): "
        f"{sorted(categorical_columns)}"
    )
    print(
        f"Kolom numerikal ({len(numeric_columns)}): "
        f"{sorted(numeric_columns)}"
    )

    print("\nJumlah nilai per kategori (diurutkan berdasarkan nilai):")
    if categorical_columns:
        for column in sorted(categorical_columns):
            counts = data[column].value_counts(dropna=False).sort_index(
                key=lambda values: values.astype(str)
            )
            print(f"\n{column}:\n{counts.to_string()}")
    else:
        print("Tidak ada kolom kategorikal.")

    print("\nRingkasan kolom numerikal (nama kolom diurutkan):")
    if numeric_columns:
        print(data[sorted(numeric_columns)].describe().to_string())
    else:
        print("Tidak ada kolom numerikal.")

    print(f"\nBaris duplikat (semua kemunculan): {len(duplicate_rows)}")
    if not duplicate_rows.empty:
        print(duplicate_rows.head(10).to_string(index=False))

    print(f"\nBaris yang memiliki null: {len(null_rows)}")
    if null_counts.empty:
        print("Jumlah null per kolom: tidak ada.")
    else:
        print("Jumlah null per kolom (terbanyak dahulu):")
        print(null_counts.to_string())
        print("Contoh baris null:")
        print(null_rows.head(10).to_string(index=False))

    print(f"\nBaris yang memiliki nilai zero numerikal: {len(zero_rows)}")
    if zero_counts.empty or zero_counts.sum() == 0:
        print("Jumlah zero per kolom: tidak ada.")
    else:
        print("Jumlah zero per kolom (terbanyak dahulu):")
        print(zero_counts[zero_counts > 0].to_string())
        print("Contoh baris zero:")
        print(zero_rows.head(10).to_string(index=False))

    slug = name.lower().replace(" ", "_")
    summary = {
        "Kategorikal": len(categorical_columns),
        "Numerikal": len(numeric_columns),
        "Duplikat": len(duplicate_rows),
        "Baris null": len(null_rows),
        "Baris zero": len(zero_rows),
    }
    plt.figure(figsize=(9, 5))
    bars = plt.bar(summary.keys(), summary.values(), color="steelblue")
    plt.title(f"Ringkasan Pemeriksaan Data - {name}")
    plt.ylabel("Jumlah")
    plt.bar_label(bars, padding=3)
    plt.tight_layout()
    output_path = BASE_DIR / f"ringkasan_{slug}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nGrafik disimpan: {output_path.name}")


def main() -> None:
    for name, path in DATASETS.items():
        data = load_dataset(name, path)
        analyze_dataset(name, data)


if __name__ == "__main__":
    main()
