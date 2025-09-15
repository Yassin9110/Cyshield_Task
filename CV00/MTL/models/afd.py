# src/models/afd.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.conv(x)

class AFDModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(AFDModule, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(kernel_size)
        # Add a final convolution to reduce the channel dimension to 1
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate channel and spatial attention logits
        channel_logits = self.ca(x)
        spatial_logits = self.sa(x)

        # Broadcast channel_logits to match spatial dimensions and combine
        combined_features = x * self.sigmoid(channel_logits) + x * self.sigmoid(spatial_logits)

        # Apply the final convolution to get a single channel attention mask
        attention_mask_logits = self.final_conv(combined_features)

        # The final mask highlights age-related features
        attention_mask = self.sigmoid(attention_mask_logits)
        return attention_mask