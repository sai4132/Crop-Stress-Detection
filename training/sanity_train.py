import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.datasets.sen12ms import SEN12MSDataset
from src.datasets.collate import collate_fn
from src.utils import paths
from src.preprocessing.transform import transform
from src.models import cnn
import time
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
import numpy as np

# --- 1. Train One Epoch Function ---
def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criterion, device):
    model.train()  # Enable dropout/batchnorm training behavior
    running_loss = 0.0
    
    for data in dataloader:
        X = data["s2"].to(device, non_blocking=True)
        y = data["label"].to(device, non_blocking=True).view(-1, 1).float()

        optimizer.zero_grad(set_to_none=True)

        y_p_logits = model(X)
        loss = criterion(y_p_logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Return the average training loss across all batches
    return running_loss / len(dataloader)


# --- 2. Validate One Epoch Function ---
def validate_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, device):
    model.eval()  # Disable dropout/freeze running batchnorm statistics
    running_loss = 0.0
    preds, labels = [], []
    
    # Disable gradient tracking to optimize Unified Memory performance
    with torch.no_grad():
        for data in dataloader:
            X = data["s2"].to(device, non_blocking=True)
            y = data["label"].to(device, non_blocking=True).view(-1, 1).float()

            y_p_logits = model(X)
            loss = criterion(y_p_logits, y)

            predictions = (torch.sigmoid(y_p_logits)>=0.5).int()
            preds.append(predictions.cpu().numpy())
            labels.append(y.cpu().numpy())

            running_loss += loss.item()

    preds, labels = np.vstack(preds).flatten(), np.vstack(labels).flatten()

    print(f"Prediction mean: {preds.mean():.2f}, min: {preds.min():.2f}, max: {preds.max():.2f}")
    print(f"Epoch Validation accuracy score: {accuracy_score(labels, preds)*100:.2f} | "
          f"precision score: {precision_score(labels, preds):.2f} | "
          f"recall score: {recall_score(labels, preds):.2f} | "
          f"F1 score: {f1_score(labels, preds):.2f}")
    print(f"Confusion matrix: \n{confusion_matrix(labels, preds)}")
    # Return the average validation loss across all batches
    return running_loss / len(dataloader)


if __name__ == "__main__":
    # Safe Device Selection
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Running pipeline execution loop on: {device}\n")

    # 1. Datasets Initialization
    train_dataset = SEN12MSDataset(
        manifest_path=paths.TRAIN_LABELS_PATH,
        transform=transform,
        sensors={"s2": [2, 3, 4,8]}
    )
    
    # Make sure you point paths.VAL_LABELS_PATH to your pre-filtered validation split index
    val_dataset = SEN12MSDataset(
        manifest_path=paths.VAL_LABELS_PATH, 
        transform=transform,
        sensors={"s2": [2, 3, 4,8]}
    )

    # 2. DataLoaders Configuration
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 3. Model Configuration
    train_channels = len(train_dataset.sensors["s2"])
    model = cnn.CNN(input_channels=train_channels).to(device)

    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # 4. Main Multi-Epoch Training Loop Engine
    TOTAL_EPOCHS = 5
    print("Starting the multi-epoch pipeline run:\n")

    for epoch in range(1, TOTAL_EPOCHS + 1):
        start_time = time.time()

        print(f"Training Epoch {epoch}")
        
        # --- Run Training Phase ---
        avg_train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )
        
        # --- Run Validation Phase ---
        avg_val_loss = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        end_time = time.time()
        
        # Time calculations
        epoch_mins = int((end_time - start_time) // 60)
        epoch_secs = int((end_time - start_time) % 60)
        
        # Comprehensive tracking logs
        print(f"Epoch {epoch:02d}/{TOTAL_EPOCHS:02d} Complete | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Duration: {epoch_mins}m {epoch_secs}s")