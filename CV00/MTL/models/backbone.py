# src/models/backbone.py

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def get_backbone(name: str, pretrained: bool = True):
    """
    Factory function to create a backbone model.

    Args:
        name (str): Name of the backbone (e.g., 'resnet50').
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        tuple: A tuple containing:
            - nn.Module: The backbone model.
            - int: The number of output features from the backbone.

    Raises:
        ValueError: If the backbone name is not supported.
    """
    if name == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)

        # Truncate the model to remove the final avgpool and fc layers
        backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        out_features = model.fc.in_features  # 2048 for ResNet50
        return backbone, out_features

    # You can easily add other backbones here using the same pattern.
    # For example, using the 'timm' library:
    #
    # if name == 'efficientnet_b0':
    #     import timm
    #     model = timm.create_model('efficientnet_b0', pretrained=pretrained, features_only=True)
    #     out_features = model.feature_info.channels(-1)
    #     return model, out_features

    else:
        raise ValueError(f"Backbone '{name}' not supported. Please use 'resnet50' or add a new one.")
