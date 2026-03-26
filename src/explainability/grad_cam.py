"""
Grad-CAM for Vision Transformer (ViT) Image Explainability
=============================================================
Generates visual explanations by computing gradient-weighted class activation
maps on ViT patch features. Shows which regions of the image the model
focuses on when making its fake/real prediction.

Adapted for transformer architectures (no convolutional feature maps).
Uses attention rollout + gradient weighting as an alternative to standard Grad-CAM.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


class MultimodalGradCAM:
    """
    Grad-CAM-like visualization for ViT within the multimodal detector.

    Since ViT has no convolutional layers, we use a combination of:
    1. Attention rollout across all ViT layers
    2. Gradient-weighted relevance from the classification loss

    This produces spatial heatmaps showing image regions most relevant
    to the model's prediction.
    """

    def __init__(self, model, device: torch.device = None, save_dir: str = "./explanations"):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def compute_attention_rollout(self, attentions: tuple, discard_ratio: float = 0.1) -> np.ndarray:
        """
        Compute attention rollout across all transformer layers.

        Attention rollout recursively multiplies attention matrices across layers
        to obtain a global attention map from the [CLS] token to all patches.

        Args:
            attentions: Tuple of attention tensors, each [B, num_heads, seq, seq]
            discard_ratio: Fraction of lowest attention weights to zero out

        Returns:
            rollout: Attention map [B, num_patches] (excluding [CLS])
        """
        result = None

        for attention in attentions:
            # Average across heads
            attn = attention.mean(dim=1)  # [B, seq, seq]

            # Discard low-attention entries
            if discard_ratio > 0:
                flat = attn.view(attn.size(0), -1)
                threshold = flat.quantile(discard_ratio, dim=1, keepdim=True)
                threshold = threshold.view(attn.size(0), 1, 1)
                attn = attn * (attn > threshold).float()

            # Add residual connection (identity matrix)
            I = torch.eye(attn.size(-1), device=attn.device).unsqueeze(0)
            attn = 0.5 * attn + 0.5 * I

            # Normalize rows
            attn = attn / attn.sum(dim=-1, keepdim=True)

            # Multiply with previous rollout
            if result is None:
                result = attn
            else:
                result = torch.bmm(attn, result)

        # Extract [CLS] → patch attention (skip [CLS] token at position 0)
        cls_attention = result[:, 0, 1:]  # [B, num_patches]
        return cls_attention.detach().cpu().numpy()

    def compute_gradient_map(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        target_class: int = None,
    ) -> dict:
        """
        Compute gradient-weighted attention map for an image.

        Args:
            input_ids: Text token IDs [1, seq_len]
            attention_mask: Text attention mask [1, seq_len]
            pixel_values: Image tensor [1, 3, H, W]
            target_class: Class to explain (None = predicted class)

        Returns:
            dict with:
                - 'heatmap': Normalized spatial heatmap [H_patches, W_patches]
                - 'prediction': Predicted class
                - 'confidence': Prediction confidence
                - 'rollout_map': Raw attention rollout [num_patches]
        """
        self.model.eval()

        # Enable gradients for this forward pass
        pixel_values = pixel_values.to(self.device).requires_grad_(True)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            mode="multimodal",
        )

        logits = outputs["logits"]
        probs = outputs["probabilities"]

        # Determine target class
        if target_class is None:
            target_class = logits.argmax(dim=-1).item()

        confidence = probs[0, target_class].item()

        # Backward pass to get gradients
        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward(retain_graph=True)

        # Get ViT attention maps
        image_attentions = outputs.get("image_attentions", None)
        if image_attentions is None:
            return {"heatmap": None, "prediction": target_class, "confidence": confidence}

        # Compute attention rollout
        rollout = self.compute_attention_rollout(image_attentions)  # [1, num_patches]
        rollout = rollout[0]  # [num_patches]

        # Compute gradient relevance
        if pixel_values.grad is not None:
            grad_magnitude = pixel_values.grad.abs().mean(dim=(0, 1))  # [H, W]
            # Resize to patch grid
            num_patches = rollout.shape[0]
            patch_grid = int(np.sqrt(num_patches))
            grad_magnitude = F.interpolate(
                grad_magnitude.unsqueeze(0).unsqueeze(0),
                size=(patch_grid, patch_grid),
                mode="bilinear",
                align_corners=False,
            ).squeeze().detach().cpu().numpy()

            # Combine rollout with gradient
            rollout_2d = rollout.reshape(patch_grid, patch_grid)
            heatmap = rollout_2d * grad_magnitude
        else:
            num_patches = rollout.shape[0]
            patch_grid = int(np.sqrt(num_patches))
            heatmap = rollout.reshape(patch_grid, patch_grid)

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return {
            "heatmap": heatmap,
            "prediction": target_class,
            "confidence": confidence,
            "rollout_map": rollout,
        }

    def visualize(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        prediction: int,
        confidence: float,
        title: str = None,
        filename: str = "gradcam_explanation.png",
        alpha: float = 0.4,
    ) -> str:
        """
        Overlay heatmap on original image and save visualization.

        Args:
            original_image: PIL Image
            heatmap: 2D numpy array [H_patch, W_patch]
            prediction: Predicted class index
            confidence: Prediction confidence
            title: Optional plot title
            filename: Output filename
            alpha: Overlay transparency

        Returns:
            Path to saved visualization
        """
        class_names = ["Real", "Fake"]

        # Resize heatmap to image dimensions
        img_array = np.array(original_image)
        h, w = img_array.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * img_array)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(img_array)
        axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(heatmap_resized, cmap="jet", interpolation="bilinear")
        axes[1].set_title("Attention Heatmap", fontsize=13, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        pred_label = class_names[prediction]
        color = "#E91E63" if prediction == 1 else "#4CAF50"
        axes[2].set_title(
            f"Prediction: {pred_label} ({confidence:.1%})",
            fontsize=13, fontweight="bold", color=color,
        )
        axes[2].axis("off")

        if title:
            fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[GRAD-CAM] Visualization saved to {save_path}")
        return str(save_path)

    def explain(
        self,
        original_image: Image.Image,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        filename: str = "gradcam_explanation.png",
    ) -> dict:
        """
        End-to-end explanation: compute heatmap and generate visualization.

        Returns:
            dict with heatmap data and path to saved visualization
        """
        result = self.compute_gradient_map(input_ids, attention_mask, pixel_values)

        if result["heatmap"] is not None:
            viz_path = self.visualize(
                original_image=original_image,
                heatmap=result["heatmap"],
                prediction=result["prediction"],
                confidence=result["confidence"],
                filename=filename,
            )
            result["visualization_path"] = viz_path

        return result
