import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import random


class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)  # output size = num_classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # raw logits
    
def to_loader(X, y, batch_size=64, shuffle=False):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def main():
    
    diag = pd.read_csv('output.csv')
    subjects = list(pd.read_csv('subjects.csv').columns)
    subjects = [subject.upper() for subject in subjects]
    
    random.seed(42)
    random.shuffle(subjects)
    labels = diag[diag['Subject'].isin(subjects)]['GroupN'].to_numpy()

    X_train = np.load("train_embeddings.npy")
    X_val = np.load("val_embeddings.npy")
    X_test = np.load("test_embeddings.npy")
    
    n = np.shape(labels)[0]
    n_train = int(0.7 * n)
    n_val   = int(0.15 * n)
    
    y_train = labels[:n_train]
    y_val   = labels[n_train:n_train + n_val]
    y_test  = labels[n_train + n_val:]
    
    
    train_loader = to_loader(X_train, y_train, batch_size=16)
    val_loader   = to_loader(X_val, y_val)
    test_loader  = to_loader(X_test, y_test)
    
    input_dim = X_train.shape[1]
    
    model = MultiLabelMLP(input_dim=input_dim, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    epochs = 500
    epoch, avg_test_loss, avg_train_loss, avg_val_loss = 0, float('inf'), float('inf'), float('inf')
    """ pbar = tqdm(range(epochs), desc=    f"Epoch {epoch+1} | "
                                        f"Train Loss: {avg_train_loss:.4f} | "
                                        f"Val Loss: {avg_val_loss:.4f}", ) """
                                    

    for epoch in range(epochs):
        # ------ Train ------
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            """ print(X_batch.shape)
            print(y_batch.shape, y_batch.dtype)  # should be (batch_size, num_labels), dtype=float
            print(y_batch[:5]) """
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)  # shape: (batch_size, num_classes)

            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss/len(train_loader)
            
        # ------ Validation ------
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu().int())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_targets).numpy()
        
        accuracy = (y_pred == y_true).mean()
        
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} Val Loss={val_loss:.4f} Acc={accuracy:.2f}")
    
        
if __name__ == '__main__':
    main()

