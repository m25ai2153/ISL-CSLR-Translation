import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.cslr_bilstm import LightweightCSLR

def train_model(
    train_loader, 
    val_loader, 
    num_classes, 
    input_dim=288, 
    epochs=50, 
    num_epochs=None, 
    lr=1e-3, 
    device="cpu", 
    **kwargs
):
    # Support both 'epochs' and 'num_epochs' argument names
    total_epochs = num_epochs if num_epochs is not None else epochs

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/best_isl_model.pth"

    # Initialize CSLR Model with dynamic input_dim
    model = LightweightCSLR(input_dim=input_dim, num_classes=num_classes).to(device)
    
    # CTC Loss (blank index = num_classes - 1)
    blank_idx = num_classes - 1
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")

    print(f"Starting training on {device}... Total Classes: {num_classes} | Input Dim: {input_dim}")

    for epoch in range(total_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets, input_lengths, target_lengths in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            # Forward pass: logits shape [batch_size, seq_len, num_classes]
            logits = model(inputs)
            
            # CTCLoss expects log_probs of shape [seq_len, batch_size, num_classes]
            log_probs = logits.transpose(0, 1)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, input_lengths, target_lengths in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                logits = model(inputs)
                log_probs = logits.transpose(0, 1)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else avg_train_loss

        print(f"Epoch [{epoch+1}/{total_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpoint saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
            print(f"--> Saved best model checkpoint to {checkpoint_path}")

    print("\nTraining complete!")
    return model