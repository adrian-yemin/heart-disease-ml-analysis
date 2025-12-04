import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pathlib import Path
from joblib import dump

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
METRICS_DIR = BASE_DIR / "results" / "metrics"
MODELS_DIR = BASE_DIR / "results" / "models"
PLOTS_DIR = BASE_DIR / "results" / "figures"
PLOTS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

print("Loading preprocessed numpy arrays...")
X_train_orig = np.load(DATA_DIR / "X_train.npy")
X_val_orig = np.load(DATA_DIR / "X_val.npy")
y_train = np.load(DATA_DIR / "y_train.npy").reshape(-1)
y_val = np.load(DATA_DIR / "y_val.npy").reshape(-1)

X_train_pca = np.load(DATA_DIR / "X_train_pca.npy")
X_val_pca = np.load(DATA_DIR / "X_val_pca.npy")

feature_sets = {
    "Original": (X_train_orig, X_val_orig),
    "PCA": (X_train_pca, X_val_pca)
}

models = {
    "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=SEED),
                           {"C": [0.01, 0.1, 1, 10, 100]}),
    "KNN": (KNeighborsClassifier(),
            {"n_neighbors": [3,5,7,9]}),
    "SVM_Linear": (SVC(kernel="linear", probability=True, random_state=SEED),
                   {"C": [0.1, 1, 10]}),
    "SVM_RBF": (SVC(kernel="rbf", probability=True, random_state=SEED),
                {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]})
}

def evaluate_model(model, X_val, y_val, model_name, feature_set_name):
    """Compute all metrics, plot confusion matrix and ROC"""
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:,1]

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds)
    rec = recall_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    roc_auc = roc_auc_score(y_val, probs)

    cm = confusion_matrix(y_val, preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix ({feature_set_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(PLOTS_DIR / f"{model_name}_{feature_set_name}_confusion.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_val, probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], "--", color="gray")
    plt.title(f"{model_name} ROC Curve ({feature_set_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(PLOTS_DIR / f"{model_name}_{feature_set_name}_roc.png")
    plt.close()

    metrics = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc
    }
    return metrics

all_results = []

for feature_set_name, (X_tr, X_val_fs) in feature_sets.items():
    print(f"\n=== Feature Set: {feature_set_name} ===")
    
    for model_name, (model, param_grid) in models.items():
        print(f"Training {model_name}...")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy")
        grid.fit(X_tr, y_train)

        best_model = grid.best_estimator_
        
        dump(best_model, MODELS_DIR / f"{model_name}_{feature_set_name}_best_model.joblib")

        metrics = evaluate_model(best_model, X_val_fs, y_val, model_name, feature_set_name)
        metrics.update({"Model": model_name, "FeatureSet": feature_set_name})
        all_results.append(metrics)

results_df = pd.DataFrame(all_results)
results_df = results_df[["Model", "FeatureSet", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
results_df.to_csv(METRICS_DIR / "classical_models_metrics.csv", index=False)
print("\n=== Classical Models Metrics ===")
print(results_df.to_string(index=False))
print("\nAll metrics saved to classical_models_metrics.csv in results/metrics/ folder")
print("Confusion matrices and ROC curves saved to results/figures/ folder")
