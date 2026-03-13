import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

def get_data_loaders(df: pd.DataFrame, batch_size: int = 32):
    # Use 'Close' as the target for demonstration
    features = df.select_dtypes(include=[np.number]).drop(columns=['Close'], errors='ignore')
    target = pd.to_numeric(df['Close'], errors='coerce')
    mask = ~target.isna()
    X = features.values[mask]
    y = target.values[mask].reshape(-1, 1).astype(np.float32)

    # Split 80/10/10
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalize using training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler for later use
    joblib.dump(scaler, "scaler.save")

    # Convert to PyTorch tensors
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    df = pd.read_csv("air_quality_cleaned.csv")
    print("Columns:", df.columns.tolist())
    print(df.head())
    loaders = get_data_loaders(df)
    print("Train/Val/Test batches:", [len(l) for l in loaders])