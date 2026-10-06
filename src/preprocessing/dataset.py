import os
import json
import torch
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, Subset

class ISLDataset(Dataset):
    """
    PyTorch Dataset wrapper for ISL keypoint features.
    """
    def __init__(self, features_dir="data/processed_features"):
        self.features_dir = features_dir
        metadata_path = os.path.join(features_dir, "dataset_metadata.json")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file '{metadata_path}' not found. Run process_dataset.py first.")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.file_list = list(self.metadata.keys())
        
        # Generate dynamic gloss-to-index mapping (Reserve Index 0 for CTC Blank)
        unique_glosses = sorted(list(set(self.metadata.values())))
        self.gloss_to_idx = {gloss: idx + 1 for idx, gloss in enumerate(unique_glosses)}
        self.idx_to_gloss = {idx: gloss for gloss, idx in self.gloss_to_idx.items()}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.features_dir, file_name)

        gloss = self.metadata[file_name]
        label = self.gloss_to_idx[gloss]

        features = np.load(file_path)  # Shape: (Num_Frames, Feature_Dim)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def pad_collate_fn(batch):
    """
    Pads variable-length frame feature sequences within a batch.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Required format for PyTorch CTC loss targets
    target_lengths = torch.ones(len(labels), dtype=torch.long)
    return padded_sequences, labels, lengths, target_lengths

def split_dataset(dataset, val_ratio=0.2, seed=42):
    """
    Splits dataset into Train and Validation sets.
    Single-sample classes are automatically forced into Train set.
    """
    np.random.seed(seed)
    gloss_counts = Counter([dataset.metadata[f] for f in dataset.file_list])

    train_indices = []
    val_indices = []

    # Group file indices by gloss
    gloss_to_indices = {}
    for idx, f in enumerate(dataset.file_list):
        gloss = dataset.metadata[f]
        gloss_to_indices.setdefault(gloss, []).append(idx)

    for gloss, indices in gloss_to_indices.items():
        # Single sample classes go 100% into Training
        if len(indices) < 2:
            train_indices.extend(indices)
        else:
            np.random.shuffle(indices)
            n_val = max(1, int(len(indices) * val_ratio))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    print(f"Dataset Split Summary:")
    print(f" - Total Samples : {len(dataset)}")
    print(f" - Train Samples : {len(train_subset)}")
    print(f" - Val Samples   : {len(val_subset)}")
    print(f" - Total Classes : {len(dataset.gloss_to_idx)}")

    return train_subset, val_subset