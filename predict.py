import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from a6_ex4 import PM_Model
from a6_ex3 import get_data_loaders
import pandas as pd

def train_model(model: nn.Module, train_loader, val_loader, test_loader, epochs=200, lr=1e-3, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    for epoch in range(epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "model.pt")

    # Test set predictions
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().flatten()
            y_pred.extend(pred)
            y_true.extend(yb.cpu().numpy().flatten())

    # Plot true vs predicted PM2.5
    plt.figure(figsize=(10, 5))
    plt.plot(y_true, label="True PM2.5")
    plt.plot(y_pred, label="Predicted PM2.5")
    plt.xlabel("Sample")
    plt.ylabel("PM2.5")
    plt.legend()
    plt.title("True vs Predicted PM2.5 on Test Set")
    plt.tight_layout()
    plt.savefig("model_prediction.pdf")
    plt.close()

if __name__ == '__main__':
    df = pd.read_csv("air_quality_cleaned.csv")
    train_loader, val_loader, test_loader = get_data_loaders(df)
    # Get feature count from the first batch
    for xb, _ in train_loader:
        input_dim = xb.shape[1]
        break
    model = PM_Model(input_dim=input_dim)
    train_model(model, train_loader, val_loader, test_loader)

