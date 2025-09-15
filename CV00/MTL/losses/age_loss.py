# src/losses/age_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeLoss(nn.Module):
    """
    Combines DEX (Deep EXpectation) loss for continuous age regression
    and standard Cross-Entropy for age group classification.
    """
    def __init__(self, num_age_bins=101):
        super(AgeLoss, self).__init__()
        self.num_age_bins = num_age_bins
        # Create a non-trainable buffer for the age values [0, 1, 2, ..., 100]
        self.register_buffer('age_range', torch.arange(0, self.num_age_bins, dtype=torch.float32))

        # We will hardcode the group mapping for clarity
        # Bins: ['10-', '11-20', '21-30', '31-40', '41-50', '51-60', '61+']
        self.group_ranges = [
            (0, 11), (11, 21), (21, 31), (31, 41),
            (41, 51), (51, 61), (61, self.num_age_bins)
        ]
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, age_logits, true_ages, true_age_groups):
        # --- 1. DEX Loss for Continuous Age ---
        softmax_probs = F.softmax(age_logits, dim=1)
        predicted_ages = (softmax_probs * self.age_range).sum(dim=1)
        loss_dex = self.mse_loss(predicted_ages, true_ages)

        # --- 2. CE Loss for Age Groups ---
        # Project 101 bins down to 7 groups by summing logits
        group_logits = torch.stack([
            age_logits[:, start:end].sum(dim=1) for start, end in self.group_ranges
        ], dim=1)

        true_age_groups_long = true_age_groups.long()
        loss_group = self.ce_loss(group_logits, true_age_groups_long)

        # Return a combined loss
        return loss_dex + loss_group