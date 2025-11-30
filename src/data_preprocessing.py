import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from ucimlrepo import fetch_ucirepo

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

def load_preprocess_save_data():
    heart = fetch_ucirepo(id=45)

    X = heart.data.features
    y = heart.data.targets

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_cols = ['cp', 'restecg', 'slope', 'ca', 'thal']

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = num_imputer.transform(X_val[numeric_cols])

    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_val[categorical_cols] = cat_imputer.transform(X_val[categorical_cols])

    X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)
    X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=False)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])

    X_train = X_train.astype(float)
    X_val = X_val.astype(float)

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    torch.save(train_loader, DATA_DIR / "train_loader.pt")
    torch.save(val_loader, DATA_DIR / "val_loader.pt")

if __name__ == "__main__":
    load_preprocess_save_data()
