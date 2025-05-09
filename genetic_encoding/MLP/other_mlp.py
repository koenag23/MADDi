import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def to_loader(X, y, batch_size=64, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    # ---- Load Labels ----
    diag = pd.read_csv('output.csv')
    subjects = list(pd.read_csv('subjects.csv').columns)
    subjects = [s.upper() for s in subjects]

    random.seed(42)
    random.shuffle(subjects)

    labels = diag[diag['Subject'].isin(subjects)]['GroupN'].to_numpy()

    # ---- Load Precomputed Embeddings ----
    X_train = np.load("train_embeddings.npy")
    X_val   = np.load("val_embeddings.npy")
    X_test  = np.load("test_embeddings.npy")

    # ---- Align Labels ----
    n_train = X_train.shape[0]
    n_val   = X_val.shape[0]
    n_test  = X_test.shape[0]

    y_train = labels[:n_train]
    y_val   = labels[n_train:n_train + n_val]
    y_test  = labels[n_train + n_val:n_train + n_val + n_test]

    # ---- DataLoaders ----
    train_loader = to_loader(X_train, y_train, batch_size=32, shuffle=True)
    val_loader   = to_loader(X_val, y_val, batch_size=32)
    test_loader  = to_loader(X_test, y_test, batch_size=32)

    # ---- Initialize MLP ----
    input_dim = X_train.shape[1]
    model = MultiLabelMLP(input_dim=input_dim, num_classes=3)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # ---- Metric Storage ----
    train_losses = []
    val_losses = []
    val_accuracies = []

    # ---- Training Loop ----
    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Acc={accuracy:.2%}")

    # ---- Test Evaluation ----
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    test_accuracy = (all_preds == all_targets).mean()
    test_f1 = f1_score(all_targets, all_preds, average='weighted')
    test_precision = precision_score(all_targets, all_preds, average='weighted')
    test_recall = recall_score(all_targets, all_preds, average='weighted')

    print(f"\nTest Accuracy: {100 * test_accuracy:.2f}%")
    print(f"F1-Score: {test_f1:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ---- Plot Loss & Accuracy ----
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.close()

if __name__ == '__main__':
    main()
