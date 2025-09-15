# src/models/aifr_model.py

import torch
import torch.nn as nn
from backbone import get_backbone
from afd import AFDModule
from heads import IdentityHead, AgeHead
from grl import GradientReversalLayer

class AIFRModel(nn.Module):
    def __init__(self, backbone_name='resnet50', embedding_size=512, num_age_bins=101, pretrained=True):
        super(AIFRModel, self).__init__()
        print(f"Initializing AIFRModel with backbone: {backbone_name}")

        # 1. Get the backbone and its output feature dimension
        self.backbone, backbone_out_features = get_backbone(backbone_name, pretrained)

        # 2. Instantiate all component modules
        self.afd = AFDModule(in_channels=backbone_out_features)
        self.identity_head = IdentityHead(in_features=backbone_out_features, embedding_size=embedding_size)

        # IMPORTANT: The AgeHead is shared between the main age prediction task and the adversarial task
        self.age_head = AgeHead(in_features=backbone_out_features, num_age_bins=num_age_bins)

        self.grl = GradientReversalLayer()

    def forward(self, image, lambda_grl=1.0):
        # --- Main Forward Pass ---
        # 1. Extract features from the backbone
        features = self.backbone(image)

        # 2. Generate the attention mask from AFD
        # This mask identifies regions/channels related to AGE
        attention_mask = self.afd(features)

        # 3. Split features into identity-focused and age-focused
        id_features = features * (1.0 - attention_mask)
        age_features = features * attention_mask

        # --- Task-Specific Paths ---
        # Path 1: Identity Recognition
        embedding = self.identity_head(id_features)

        # Path 2: Age Prediction
        age_logits = self.age_head(age_features)

        # Path 3: Adversarial Age Invariance
        reversed_id_features = self.grl(id_features, lambda_grl)
        adversarial_age_logits = self.age_head(reversed_id_features)

        # Return a dictionary for clarity in the training loop
        return {
            "embedding": embedding,
            "age_logits": age_logits,
            "adversarial_age_logits": adversarial_age_logits,
            "attention_mask": attention_mask  # Useful for visualization
        }

# --- Sanity Check ---
if __name__ == '__main__':
    print("--- Running Sanity Check for AIFRModel ---")

    # Easily switch backbones here to test
    # model = AIFRModel(backbone_name='resnet50')
    # model = AIFRModel(backbone_name='efficientnet_b0') # Would require 'timm'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AIFRModel(backbone_name='resnet50').to(device)
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(4, 3, 112, 112).to(device) # Batch size of 4

    with torch.no_grad():
        outputs = model(dummy_input, lambda_grl=0.5)

    print("\n--- Model Initialized Successfully ---")
    print(f"Backbone: {model.backbone.__class__.__name__}")
    print("\n--- Output Shapes ---")
    for name, tensor in outputs.items():
        print(f"{name:<25}: {tensor.shape}")

    # Check output shapes
    assert outputs['embedding'].shape == (4, 512)
    assert outputs['age_logits'].shape == (4, 101)
    assert outputs['adversarial_age_logits'].shape == (4, 101)
    # For resnet50, features after layer4 on a 112x112 input are 4x4
    assert outputs['attention_mask'].shape == (4, 1, 4, 4)
    print("\n✅ Sanity check passed!")