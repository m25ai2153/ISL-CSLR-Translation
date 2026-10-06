import torch
import numpy as np

class CTCDecoder:
    """
    Decodes frame-level output probabilities from the model into ISL gloss sequences using CTC greedy decoding.
    """
    def __init__(self, idx_to_gloss, blank_idx=0):
        self.idx_to_gloss = idx_to_gloss
        self.blank_idx = blank_idx

    def decode_greedy(self, logits):
        """
        Applies greedy decoding to a single sequence's logits tensor.
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        # Get top class prediction for each frame timestep
        predictions = np.argmax(logits, axis=-1)

        # Collapse duplicate consecutive predictions and omit CTC blank token
        decoded_ids = []
        prev_id = None

        for p in predictions:
            if p != prev_id:
                if p != self.blank_idx:
                    decoded_ids.append(p)
                prev_id = p

        # Map integer indices back to sign gloss words
        gloss_sequence = [self.idx_to_gloss.get(idx, "<UNK>") for idx in decoded_ids]
        return gloss_sequence