"""
Vision Transformer (ViT) Image Encoder for Fake News Detection
===============================================================
Extracts patch-level and global image representations using a pretrained ViT.
Supports partial layer freezing and outputs compatible with cross-modal fusion.
"""

import torch
import torch.nn as nn
from transformers import ViTModel
from torchvision import transforms


class ViTImageEncoder(nn.Module):
    """
    ViT-based encoder that extracts patch-level and [CLS]-level
    representations from input images.

    Args:
        model_name (str): HuggingFace ViT model identifier.
        hidden_size (int): ViT hidden dimension (typically 768).
        image_size (int): Expected input image size.
        freeze_layers (int): Number of initial encoder layers to freeze.
        projection_dim (int): Output projection dimension for fusion compatibility.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        hidden_size: int = 768,
        image_size: int = 224,
        freeze_layers: int = 8,
        projection_dim: int = 512,
    ):
        super().__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size

        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(model_name, output_attentions=True)

        # Freeze early layers
        self._freeze_layers(freeze_layers)

        # Project ViT hidden size to fusion dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Image preprocessing pipeline (for training with augmentation)
        self.train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.eval_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _freeze_layers(self, num_layers: int):
        """Freeze the embedding layer and the first `num_layers` encoder layers."""
        # Freeze patch embeddings
        for param in self.vit.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified encoder layers
        for i in range(min(num_layers, len(self.vit.encoder.layer))):
            for param in self.vit.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> dict:
        """
        Forward pass through ViT encoder.

        Args:
            pixel_values: Preprocessed image tensor [batch_size, 3, H, W]

        Returns:
            dict with keys:
                - 'patch_output': Projected patch-level features [B, num_patches+1, proj_dim]
                - 'pooled_output': Projected [CLS] representation [B, proj_dim]
                - 'attentions': Tuple of attention weights from each layer
        """
        outputs = self.vit(pixel_values=pixel_values)

        # Patch-level representations (includes [CLS] token at position 0)
        sequence_output = outputs.last_hidden_state            # [B, num_patches+1, 768]
        projected_sequence = self.projection(sequence_output)  # [B, num_patches+1, proj_dim]

        # [CLS] token representation
        cls_output = sequence_output[:, 0, :]                   # [B, 768]
        projected_cls = self.projection(cls_output)             # [B, proj_dim]

        return {
            "patch_output": projected_sequence,
            "pooled_output": projected_cls,
            "attentions": outputs.attentions,
        }

    def get_transform(self, train: bool = True) -> transforms.Compose:
        """Return the appropriate image transform pipeline."""
        return self.train_transform if train else self.eval_transform
