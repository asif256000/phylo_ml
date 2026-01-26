import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset labels and basic sanity checks")
    parser.add_argument(
        "--dataset",
        "-d",
        default="/home/asifiqbal/anton_test/phylo_simulator/four_taxa/iqtree/3.3b/U0-0.1/npy_data/U0_01_m33b_4t_30k.npy",
        help="Path to the .npy dataset file",
    )
    args = parser.parse_args()

    # Load dataset
    data = np.load(args.dataset)

    # Check topology labels (one-hot encoded)
    y_top = data["y_top"]
    print(f"y_top shape: {y_top.shape}")
    print(f"y_top dtype: {y_top.dtype}")
    print(f"Unique raw values: {np.unique(y_top)}")

    # Validate one-hot rows
    row_sums = y_top.sum(axis=1)
    print(f"Row sum min/max: {row_sums.min()} / {row_sums.max()}")
    invalid_rows = np.where(~np.isclose(row_sums, 1.0))[0]
    print(f"Non one-hot rows: {len(invalid_rows)}")

    # Convert to class ids via argmax for distribution
    labels = np.argmax(y_top, axis=1)
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution (via argmax):")
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples ({cnt/len(labels)*100:.2f}%)")

    # Check for invalid values
    print(f"\nMin value: {y_top.min()}, Max value: {y_top.max()}")
    print(f"Any NaN: {np.isnan(y_top).any()}")
    print(f"Any negative: {(y_top < 0).any()}")

    # Sample check
    print("\nFirst 20 one-hot rows:")
    print(y_top[:20])
    print("\nFirst 20 class ids:")
    print(labels[:20])

    # Check if labels match expected classes (0-4)
    expected_classes = set(range(5))
    actual_classes = set(np.unique(labels).tolist())
    if actual_classes != expected_classes:
        print(f"\n⚠️ WARNING: Missing classes: {expected_classes - actual_classes}")
        print(f"⚠️ WARNING: Unexpected classes: {actual_classes - expected_classes}")

    # Check branch labels consistency
    y_br = data["y_br"]
    print(f"\n\nBranch labels (y_br) shape: {y_br.shape}")
    print("Sample y_br values (first 3):")
    print(y_br[:3])


if __name__ == "__main__":
    main()
