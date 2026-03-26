"""
Multimodal Fake News Detector
==============================
End-to-end model integrating:
    1. BertTextEncoder   → Contextual text features
    2. ViTImageEncoder   → Patch-level image features
    3. CrossModalAttentionFusion → Bidirectional cross-modal attention + gated fusion
    4. Classification Head → Fake/Real prediction

Supports both multimodal and unimodal (text-only / image-only) inference
for ablation studies and comparison experiments.
"""

import torch
import torch.nn as nn

from .text_encoder import BertTextEncoder
from .image_encoder import ViTImageEncoder
from .cross_modal_attention import CrossModalAttentionFusion


class MultimodalFakeNewsDetector(nn.Module):
    """
    Full multimodal fake news detection model.

    Architecture:
        Text  →  BERT Encoder  →  projected text tokens  ─┐
                                                            ├→ Cross-Modal Attention → Gated Fusion → Classifier
        Image →  ViT Encoder   →  projected image patches ─┘
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        text_cfg = config["model"]["text_encoder"]
        image_cfg = config["model"]["image_encoder"]
        fusion_cfg = config["model"]["fusion"]
        cls_cfg = config["model"]["classifier"]

        # --- Modality Encoders ---
        self.text_encoder = BertTextEncoder(
            model_name=text_cfg["name"],
            hidden_size=text_cfg["hidden_size"],
            max_length=text_cfg["max_length"],
            freeze_layers=text_cfg["freeze_layers"],
            projection_dim=fusion_cfg["hidden_size"],
        )

        self.image_encoder = ViTImageEncoder(
            model_name=image_cfg["name"],
            hidden_size=image_cfg["hidden_size"],
            image_size=image_cfg["image_size"],
            freeze_layers=image_cfg["freeze_layers"],
            projection_dim=fusion_cfg["hidden_size"],
        )

        # --- Cross-Modal Fusion ---
        self.fusion = CrossModalAttentionFusion(
            hidden_size=fusion_cfg["hidden_size"],
            num_heads=fusion_cfg["num_heads"],
            num_layers=fusion_cfg["num_cross_attn_layers"],
            dropout=fusion_cfg["dropout"],
        )

        # --- Classification Head ---
        classifier_layers = []
        in_dim = fusion_cfg["hidden_size"]
        for hidden_dim in cls_cfg["hidden_sizes"]:
            classifier_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(cls_cfg["dropout"]),
            ])
            in_dim = hidden_dim
        classifier_layers.append(nn.Linear(in_dim, cls_cfg["num_classes"]))
        self.classifier = nn.Sequential(*classifier_layers)

        # --- Unimodal classifiers for ablation ---
        self.text_only_classifier = nn.Sequential(
            nn.Linear(fusion_cfg["hidden_size"], cls_cfg["hidden_sizes"][-1]),
            nn.GELU(),
            nn.Dropout(cls_cfg["dropout"]),
            nn.Linear(cls_cfg["hidden_sizes"][-1], cls_cfg["num_classes"]),
        )

        self.image_only_classifier = nn.Sequential(
            nn.Linear(fusion_cfg["hidden_size"], cls_cfg["hidden_sizes"][-1]),
            nn.GELU(),
            nn.Dropout(cls_cfg["dropout"]),
            nn.Linear(cls_cfg["hidden_sizes"][-1], cls_cfg["num_classes"]),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        mode: str = "multimodal",
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids:      [B, seq_len]
            attention_mask: [B, seq_len]
            pixel_values:   [B, 3, H, W]
            token_type_ids: [B, seq_len] (optional)
            mode: 'multimodal' | 'text_only' | 'image_only'

        Returns:
            dict with:
                - 'logits': Classification logits [B, num_classes]
                - 'probabilities': Softmax probabilities [B, num_classes]
                - 'fused_features': Fused representation (multimodal mode)
                - 'text_attentions': BERT attention weights
                - 'image_attentions': ViT attention weights
                - 'cross_attention_maps': Cross-modal attention weights
        """
        result = {}

        if mode == "text_only":
            text_out = self.text_encoder(input_ids, attention_mask, token_type_ids)
            logits = self.text_only_classifier(text_out["pooled_output"])
            result["logits"] = logits
            result["text_attentions"] = text_out["attentions"]

        elif mode == "image_only":
            image_out = self.image_encoder(pixel_values)
            logits = self.image_only_classifier(image_out["pooled_output"])
            result["logits"] = logits
            result["image_attentions"] = image_out["attentions"]

        else:  # multimodal (default)
            # Encode both modalities
            text_out = self.text_encoder(input_ids, attention_mask, token_type_ids)
            image_out = self.image_encoder(pixel_values)

            # Cross-modal fusion
            fusion_out = self.fusion(
                text_features=text_out["sequence_output"],
                image_features=image_out["patch_output"],
                text_mask=attention_mask,
            )

            # Classification
            logits = self.classifier(fusion_out["fused"])

            result["logits"] = logits
            result["fused_features"] = fusion_out["fused"]
            result["text_enriched"] = fusion_out["text_enriched"]
            result["image_enriched"] = fusion_out["image_enriched"]
            result["text_attentions"] = text_out["attentions"]
            result["image_attentions"] = image_out["attentions"]
            result["cross_attention_maps"] = fusion_out["cross_attention_maps"]

        result["probabilities"] = torch.softmax(logits, dim=-1)
        return result

    def get_trainable_params(self) -> dict:
        """Return a summary of trainable vs frozen parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "trainable_pct": 100.0 * trainable / total if total > 0 else 0,
        }
