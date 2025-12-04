import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

def sanity_check():
    print("\n=========== LOADING ARRAYS ===========")
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")

    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("y_train:", y_train.shape)
    print("y_val:", y_val.shape)

    print("\n=========== CHECK #1: MATCHING LENGTHS ===========")
    if len(X_train) != len(y_train):
        print("ERROR: X_train and y_train lengths differ!")
    else:
        print("✔ Train lengths match")

    if len(X_val) != len(y_val):
        print("ERROR: X_val and y_val lengths differ!")
    else:
        print("✔ Val lengths match")

    total_samples = len(X_train) + len(X_val)
    train_ratio = len(X_train) / total_samples
    print(f"Train ratio: {train_ratio:.3f}  (expected ≈ 0.80)")

    print("\n=========== CHECK #2: NaN DETECTION ===========")
    print("X_train has NaNs:", np.isnan(X_train).any())
    print("X_val has NaNs:", np.isnan(X_val).any())
    print("y_train has NaNs:", np.isnan(y_train).any())
    print("y_val has NaNs:", np.isnan(y_val).any())

    print("\n=========== CHECK #3: DTYPE ===========")
    print("X_train dtype:", X_train.dtype)
    print("y_train dtype:", y_train.dtype)
    print("X_val dtype:", X_val.dtype)
    print("y_val dtype:", y_val.dtype)

    if X_train.dtype != np.float64 and X_train.dtype != np.float32:
        print("WARNING: X_train should be float!")
    if y_train.dtype not in [np.int32, np.int64]:
        print("WARNING: y_train should be integer labels!")

    print("\n=========== CHECK #4: LABEL DISTRIBUTION ===========")
    unique, counts = np.unique(y_train, return_counts=True)
    print("Train labels:", dict(zip(unique, counts)))

    unique, counts = np.unique(y_val, return_counts=True)
    print("Val labels:", dict(zip(unique, counts)))

    print("\n=========== CHECK #5: BASIC STATS OF SCALED NUMERIC FEATURES ===========")
    # assume first 5 numeric columns (age, trestbps, chol, thalach, oldpeak)
    num_cols = list(range(5))

    train_means = X_train[:, num_cols].mean(axis=0)
    train_stds = X_train[:, num_cols].std(axis=0)

    print("Numeric column means (should be near 0):", train_means)
    print("Numeric column stds  (should be near 1):", train_stds)

    print("\n=========== CHECK #6: SAMPLE ROW ===========")
    print("X_train[0]:\n", X_train[0])
    print("y_train[0]:", y_train[0])

    print("\n=========== ALL CHECKS FINISHED ===========")

if __name__ == "__main__":
    sanity_check()
