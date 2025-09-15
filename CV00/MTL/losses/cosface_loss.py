# src/losses/cosface_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFaceLoss(nn.Module):
    """
    Large Margin Cosine Loss (CosFace).

    Args:
        embedding_size (int): Dimension of the embedding vectors.
        num_classes (int): Number of identity classes.
        margin (float): Margin penalty.
        scale (float): Scaling factor.
    """
    def __init__(self, embedding_size, num_classes, margin=0.35, scale=64.0):
        super(CosFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.m = margin
        self.s = scale

        # This is the weight matrix for the final classifier
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # L2-normalize embeddings and weight matrix
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights_normalized = F.normalize(self.weight, p=2, dim=1)

        # Calculate cosine similarity (dot product)
        cosine_sim = F.linear(embeddings, weights_normalized)

        # Create a one-hot vector for the labels
        one_hot = torch.zeros_like(cosine_sim)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply the margin penalty to the correct class
        # phi = cos(theta) - m
        output = (one_hot * (cosine_sim - self.m)) + ((1.0 - one_hot) * cosine_sim)

        # Scale the logits
        output *= self.s

        labels_long = labels.long()
        # Calculate standard cross-entropy loss
        return F.cross_entropy(output, labels_long)