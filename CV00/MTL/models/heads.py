# src/models/heads.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityHead(nn.Module):
    def __init__(self, in_features, embedding_size=512):
        super(IdentityHead, self).__init__()
        self.fc = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn(x)

        # L2-normalize the embeddings
        embedding = F.normalize(x, p=2, dim=1)
        return embedding

class AgeHead(nn.Module):
    def __init__(self, in_features, num_age_bins=101):
        super(AgeHead, self).__init__()
        self.fc_stack = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_age_bins)
        )

    def forward(self, x):
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        age_logits = self.fc_stack(x)
        return age_logits