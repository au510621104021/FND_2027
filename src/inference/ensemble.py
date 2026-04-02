"""Utilities for combining predictions from multiple checkpoints."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

from PIL import Image

from .predictor import MultimodalPredictor


class EnsemblePredictor:
    """Averaging ensemble over multiple multimodal predictors."""

    def __init__(self, predictors: Sequence[MultimodalPredictor], weights: Optional[Sequence[float]] = None):
        if not predictors:
            raise ValueError("At least one predictor is required.")
        self.predictors = list(predictors)
        if weights is None:
            self.weights = [1.0 / len(self.predictors)] * len(self.predictors)
        else:
            if len(weights) != len(self.predictors):
                raise ValueError("weights length must match number of predictors")
            total = float(sum(weights))
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            self.weights = [float(w) / total for w in weights]

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: Sequence[str],
        device=None,
        weights: Optional[Sequence[float]] = None,
    ) -> "EnsemblePredictor":
        predictors = [MultimodalPredictor.from_checkpoint(path, device=device) for path in checkpoints]
        return cls(predictors, weights=weights)

    def predict(
        self,
        text: str,
        image: Union[str, Image.Image, None] = None,
        mode: str = "multimodal",
        threshold: float = 0.5,
    ) -> dict:
        outputs = [p.predict(text=text, image=image, mode=mode, threshold=threshold) for p in self.predictors]

        avg_real = sum(weight * out["probabilities"]["real"] for weight, out in zip(self.weights, outputs))
        avg_fake = sum(weight * out["probabilities"]["fake"] for weight, out in zip(self.weights, outputs))

        pred_class = 1 if avg_fake >= avg_real else 0
        if pred_class == 1 and avg_fake < threshold:
            pred_class = 0

        confidence = avg_fake if pred_class == 1 else avg_real

        return {
            "prediction": "Fake" if pred_class == 1 else "Real",
            "predicted_class": pred_class,
            "confidence": float(confidence),
            "probabilities": {
                "real": float(avg_real),
                "fake": float(avg_fake),
            },
            "individual_results": outputs,
        }
