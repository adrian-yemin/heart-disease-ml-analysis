import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random
from CNN import CNN
import seaborn as sns

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

X_train = np.load(DATA_DIR / "X_train.npy")
X_val = np.load(DATA_DIR / "X_val.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
y_val = np.load(DATA_DIR / "y_val.npy")
X_train_pca = np.load(DATA_DIR / "X_train_pca.npy")
X_val_pca = np.load(DATA_DIR / "X_val_pca.npy")

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1,1)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1,1)
X_train_pca_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
X_val_pca_tensor = torch.tensor(X_val_pca, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_pca_dataset = TensorDataset(X_train_pca_tensor, y_train_tensor)
val_pca_dataset = TensorDataset(X_val_pca_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

train_pca_loader = DataLoader(train_pca_dataset, batch_size=32, shuffle=True)
val_pca_loader = DataLoader(val_pca_dataset, batch_size=32)

def train_model(model, train_loader, val_loader, input_dim, dataset_name, device, num_epochs=20, lr=0.001):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds_prob = torch.sigmoid(outputs).cpu().numpy()
            preds_bin = (preds_prob >= 0.5).astype(int)
            all_preds.extend(preds_bin)
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, [float(p) for p in all_preds])
    }

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({dataset_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(FIGURES_DIR / f"cnn_confusion_matrix_{dataset_name}.png")
    plt.close()

    fpr, tpr, thresholds = roc_curve(all_labels, [float(p) for p in all_preds])
    roc_auc_val = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_val:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.title(f"ROC Curve ({dataset_name})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(FIGURES_DIR / f"cnn_roc_curve_{dataset_name}.png")
    plt.close()

    torch.save(model.state_dict(), MODELS_DIR / f"cnn_model_{dataset_name}.pth")
    pd.DataFrame([metrics]).to_csv(METRICS_DIR / f"cnn_metrics_{dataset_name}.csv", index=False)
    print(f"\n--- {dataset_name} Metrics ---")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")
    print("---------------------------\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim_orig = X_train.shape[1]
input_dim_pca = X_train_pca.shape[1]

model_orig = CNN(input_dim_orig)
train_model(model_orig, train_loader, val_loader, input_dim_orig, "original", device)

model_pca = CNN(input_dim_pca)
train_model(model_pca, train_pca_loader, val_pca_loader, input_dim_pca, "pca", device)
