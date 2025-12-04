import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

def run_pca_analysis(n_components=None):
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")

    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_val_pca = pca.transform(X_val)

    np.save(DATA_DIR / "X_train_pca.npy", X_train_pca)
    np.save(DATA_DIR / "X_val_pca.npy", X_val_pca)

    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(exp_var)+1), exp_var, marker="o")
    plt.title("PCA Scree Plot (Variance Explained)")
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_scree_plot.png")
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(cum_var)+1), cum_var, marker="o")
    plt.title("Cumulative Variance Explained")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.grid()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_cumulative_variance.png")
    plt.close()

    if X_train_pca.shape[1] >= 2:
        plt.figure(figsize=(8,6))
        for label in np.unique(y_train):
            idx = y_train == label
            plt.scatter(
                X_train_pca[idx, 0],
                X_train_pca[idx, 1],
                label=f"Class {label}",
                alpha=0.7
            )
        plt.title("PCA Projection (Train Set)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pca_2d_scatter.png")
        plt.close()

if __name__ == "__main__":
    run_pca_analysis(n_components=10)
