import os
import torch
import numpy as np
from torch.utils.data import Dataset

class ISLDataset(Dataset):
    """
    PyTorch Dataset wrapper for loaded ISL keypoint numpy files.
    """
    def __init__(self, features_dir, label_map):
        self.features_dir = features_dir
        self.label_map = label_map
        self.file_list = [f for f in os.listdir(features_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.features_dir, file_name)

        # Extract target gloss name from filename prefix (e.g., HEADACHE_s01_v01.npy -> HEADACHE)
        gloss_name = file_name.split('_')[0].upper()
        label = self.label_map.get(gloss_name, 0)

        features = np.load(file_path)  # Shape: (Num_Frames, Feature_Dim)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def pad_collate_fn(batch):
    """
    Pads variable-length frame feature sequences within a batch for BiLSTM processing.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_sequences, labels, lengths