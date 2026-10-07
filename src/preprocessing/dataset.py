import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, Subset

class ISLDataset(Dataset):
    def __init__(self, feature_dir="data/processed_features"):
        self.feature_dir = feature_dir
        self.metadata_path = os.path.join(feature_dir, "dataset_metadata.json")
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_path}")
            
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)  # dict: {filename.npy: gloss_str}
            
        self.file_list = list(self.metadata.keys())
        
        # Build class index mapping
        unique_glosses = sorted(list(set(self.metadata.values())))
        self.label_to_idx = {gloss: idx for idx, gloss in enumerate(unique_glosses)}
        self.idx_to_label = {idx: gloss for gloss, idx in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.file_list)
        
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        file_path = os.path.join(self.feature_dir, filename)
        
        feature = np.load(file_path)  # Shape: (T, 288)
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        
        gloss = self.metadata[filename]
        label_idx = self.label_to_idx[gloss]
        
        return feature_tensor, label_idx


def pad_collate_fn(batch):
    """
    Pads dynamic frame sequences in a batch to the longest sequence in that batch.
    """
    features, labels = zip(*batch)
    
    input_lengths = torch.tensor([f.size(0) for f in features], dtype=torch.long)
    target_lengths = torch.ones(len(labels), dtype=torch.long)
    
    padded_features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    targets = torch.tensor(labels, dtype=torch.long)
    
    return padded_features, targets, input_lengths, target_lengths


def split_dataset_single_sample_aware(dataset, test_size=0.15, seed=42):
    """
    Splits ISLDataset into Train and Validation subsets safely.
    Classes with only 1 sample are automatically routed to Train set to prevent stratification errors.
    Also saves checkpoints/label_map.json.
    """
    os.makedirs("checkpoints", exist_ok=True)
    
    class_indices = {}
    for idx, filename in enumerate(dataset.file_list):
        gloss = dataset.metadata[filename]
        label_idx = dataset.label_to_idx[gloss]
        if label_idx not in class_indices:
            class_indices[label_idx] = []
        class_indices[label_idx].append(idx)
        
    train_indices = []
    val_indices = []
    
    np.random.seed(seed)
    
    for label_idx, indices in class_indices.items():
        if len(indices) == 1:
            train_indices.extend(indices)
        else:
            n_val = max(1, int(len(indices) * test_size))
            if n_val >= len(indices):
                n_val = len(indices) - 1
            shuffled = np.random.permutation(indices)
            val_indices.extend(shuffled[:n_val])
            train_indices.extend(shuffled[n_val:])
            
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    num_classes = len(dataset.label_to_idx)
    
    label_map_path = os.path.join("checkpoints", "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump(dataset.label_to_idx, f, indent=4)
        
    return train_set, val_set, num_classes, dataset.label_to_idx