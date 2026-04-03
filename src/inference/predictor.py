"""
Real-time Multimodal Prediction Engine
========================================
Provides a clean inference API for single-sample and batch predictions.
Handles model loading, preprocessing, prediction, and explainability
in a unified pipeline.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, List

from ..models.multimodal_detector import MultimodalFakeNewsDetector
from ..data.preprocessing import TextPreprocessor, ImagePreprocessor
from ..explainability.grad_cam import MultimodalGradCAM
from ..explainability.attention_viz import TextAttentionVisualizer
from transformers import BertTokenizer


class MultimodalPredictor:
    """
    High-level prediction interface for the multimodal fake news detector.

    Usage:
        predictor = MultimodalPredictor.from_checkpoint("best_model.pt")
        result = predictor.predict(text="Breaking news...", image_path="photo.jpg")
        result = predictor.predict_with_explanation(text="...", image_path="...")
    """

    CLASS_NAMES = ["Real", "Fake"]

    def __init__(
        self,
        model: MultimodalFakeNewsDetector,
        config: dict,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        self.config = config

        text_cfg = config["model"]["text_encoder"]
        image_cfg = config["model"]["image_encoder"]

        # Preprocessing
        self.text_preprocessor = TextPreprocessor()
        self.image_preprocessor = ImagePreprocessor(image_size=image_cfg["image_size"])
        self.tokenizer = BertTokenizer.from_pretrained(text_cfg["name"])
        self.max_length = text_cfg["max_length"]

        # Explainability modules
        explainability_cfg = config.get("explainability", {})
        self.grad_cam = MultimodalGradCAM(
            model=self.model,
            device=self.device,
            save_dir=explainability_cfg.get("save_dir", "./explanations"),
        )
        self.text_viz = TextAttentionVisualizer(
            tokenizer_name=text_cfg["name"],
            save_dir=explainability_cfg.get("save_dir", "./explanations"),
            layer_index=explainability_cfg.get("text_attention", {}).get("layer_index", -1),
            head_aggregation=explainability_cfg.get("text_attention", {}).get("head_aggregation", "mean"),
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: torch.device = None,
    ) -> "MultimodalPredictor":
        """
        Load a predictor from a saved checkpoint.

        Args:
            checkpoint_path: Path to .pt checkpoint file
            device: Target device

        Returns:
            Initialized MultimodalPredictor
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        config = checkpoint["config"]
        model = MultimodalFakeNewsDetector(config)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"[PREDICTOR] Model loaded from {checkpoint_path}")
        if "metrics" in checkpoint:
            print(f"  Checkpoint metrics: {checkpoint['metrics']}")

        return cls(model=model, config=config, device=device)

    def _preprocess_text(self, text: str) -> dict:
        """Preprocess and tokenize text."""
        clean_text = self.text_preprocessor(text)
        encoded = self.tokenizer(
            clean_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """Load and preprocess image."""
        if isinstance(image, str):
            image = self.image_preprocessor.load_image(image)
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a file path or PIL Image")

        tensor = self.image_preprocessor(image, train=False)
        return tensor.unsqueeze(0).to(self.device)  # [1, 3, H, W]

    @torch.no_grad()
    def predict(
        self,
        text: str,
        image: Union[str, Image.Image] = None,
        mode: str = "multimodal",
        threshold: float = 0.5,
    ) -> dict:
        """
        Make a single prediction.

        Args:
            text: Input text string
            image: Image file path or PIL Image (optional for text_only mode)
            mode: 'multimodal' | 'text_only' | 'image_only'
            threshold: Classification threshold

        Returns:
            dict with prediction, confidence, probabilities
        """
        self.model.eval()

        # Preprocess
        text_inputs = self._preprocess_text(text)

        if image is not None:
            pixel_values = self._preprocess_image(image)
        else:
            pixel_values = self.image_preprocessor.get_blank_tensor().unsqueeze(0).to(self.device)
            if mode == "multimodal":
                mode = "text_only"  # Fallback if no image provided

        # Forward pass
        outputs = self.model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=pixel_values,
            token_type_ids=text_inputs.get("token_type_ids"),
            mode=mode,
        )

        probs = outputs["probabilities"][0].cpu().numpy()
        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])

        # Apply threshold (for binary: if fake_prob < threshold, classify as real)
        if pred_class == 1 and probs[1] < threshold:
            pred_class = 0
            confidence = float(probs[0])

        return {
            "prediction": self.CLASS_NAMES[pred_class],
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": {
                "real": float(probs[0]),
                "fake": float(probs[1]),
            },
        }

    def predict_with_explanation(
        self,
        text: str,
        image: Union[str, Image.Image] = None,
        filename_prefix: str = "sample",
    ) -> dict:
        """
        Make a prediction with full explainability (Grad-CAM + text attention).

        Args:
            text: Input text
            image: Image path or PIL Image
            filename_prefix: Prefix for saved explanation files

        Returns:
            dict with prediction AND explanation paths/data
        """
        self.model.eval()

        text_inputs = self._preprocess_text(text)
        original_image = None

        if image is not None:
            if isinstance(image, str):
                original_image = self.image_preprocessor.load_image(image)
            else:
                original_image = image
            pixel_values = self._preprocess_image(original_image)
        else:
            pixel_values = self.image_preprocessor.get_blank_tensor().unsqueeze(0).to(self.device)

        # Forward pass WITH gradients (needed for Grad-CAM)
        outputs = self.model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            pixel_values=pixel_values,
            token_type_ids=text_inputs.get("token_type_ids"),
            mode="multimodal",
        )

        probs = outputs["probabilities"][0].detach().cpu().numpy()
        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])

        result = {
            "prediction": self.CLASS_NAMES[pred_class],
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": {
                "real": float(probs[0]),
                "fake": float(probs[1]),
            },
            "explanations": {},
        }

        # --- Grad-CAM on image ---
        if original_image is not None:
            try:
                gradcam_result = self.grad_cam.explain(
                    original_image=original_image,
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    pixel_values=pixel_values,
                    filename=f"{filename_prefix}_gradcam.png",
                )
                result["explanations"]["grad_cam"] = gradcam_result
            except Exception as e:
                print(f"[WARNING] Grad-CAM failed: {e}")

        # --- Text attention visualization ---
        try:
            text_explanation = self.text_viz.explain(
                model_outputs=outputs,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                filename_prefix=filename_prefix,
            )
            result["explanations"]["text_attention"] = text_explanation
        except Exception as e:
            print(f"[WARNING] Text attention visualization failed: {e}")

        return result

    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        images: List[Union[str, Image.Image]] = None,
        mode: str = "multimodal",
        batch_size: int = 16,
    ) -> List[dict]:
        """
        Batch prediction for multiple samples.

        Args:
            texts: List of input texts
            images: List of image paths/PIL Images (optional)
            mode: Inference mode
            batch_size: Processing batch size

        Returns:
            List of prediction dicts
        """
        self.model.eval()
        results = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_images = images[i:i + batch_size] if images else [None] * len(batch_texts)

            # Tokenize batch
            clean_texts = [self.text_preprocessor(t) for t in batch_texts]
            encoded = self.tokenizer(
                clean_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
            token_type_ids = encoded.get("token_type_ids", torch.zeros_like(input_ids)).to(self.device)

            # Process images
            pixel_tensors = []
            for img in batch_images:
                if img is not None:
                    pixel_tensors.append(self._preprocess_image(img).squeeze(0))
                else:
                    pixel_tensors.append(self.image_preprocessor.get_blank_tensor().to(self.device))
            pixel_values = torch.stack(pixel_tensors)

            # Forward
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                mode=mode,
            )

            probs = outputs["probabilities"].cpu().numpy()
            for j in range(len(batch_texts)):
                pred_class = int(probs[j].argmax())
                results.append({
                    "prediction": self.CLASS_NAMES[pred_class],
                    "predicted_class": pred_class,
                    "confidence": float(probs[j][pred_class]),
                    "probabilities": {
                        "real": float(probs[j][0]),
                        "fake": float(probs[j][1]),
                    },
                })

        return results
