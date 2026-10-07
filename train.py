import os
import torch
from torch.utils.data import DataLoader
from src.preprocessing.dataset import ISLDataset, pad_collate_fn, split_dataset_single_sample_aware
from src.models.trainer import train_model

def main():
    feature_dir = "data/processed_features"
    
    # Load dataset and build single-sample aware train/val split
    full_dataset = ISLDataset(feature_dir=feature_dir)
    train_dataset, val_dataset, num_classes, label_map = split_dataset_single_sample_aware(full_dataset)

    print("Dataset Split Summary:")
    print(f" - Total Samples : {len(full_dataset)}")
    print(f" - Train Samples : {len(train_dataset)}")
    print(f" - Val Samples   : {len(val_dataset)}")
    print(f" - Total Classes : {num_classes}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        collate_fn=pad_collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        collate_fn=pad_collate_fn,
        num_workers=0
    )

    # Train Model (input_dim=288, num_classes + 1 for CTC blank token)
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes + 1,  # CTC blank token added at end
        input_dim=288,
        epochs=50,
        lr=1e-3,
        device="cpu"
    )

if __name__ == "__main__":
    main()