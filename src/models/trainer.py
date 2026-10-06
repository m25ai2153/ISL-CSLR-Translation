import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from cslr_bilstm import LightweightCSLR

def train_model(train_loader, val_loader, input_dim, num_classes, epochs=30, lr=1e-3, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_isl_model.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = LightweightCSLR(input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch, x_lens, y_lens in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)  # (B, T, C)
            log_probs = logits.log_softmax(2).transpose(0, 1)  # (T, B, C) required for CTC

            loss = criterion(log_probs, y_batch, x_lens, y_lens)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # Validation phase
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Overwrite checkpoint only when validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, best_checkpoint_path)
            print(f"--> Saved best checkpoint: {best_checkpoint_path}")

def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch, x_lens, y_lens in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            log_probs = logits.log_softmax(2).transpose(0, 1)
            loss = criterion(log_probs, y_batch, x_lens, y_lens)
            val_loss += loss.item()
    return val_loss / max(len(val_loader), 1)