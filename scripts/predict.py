"""
CLI Prediction Script
=======================
Quick command-line prediction with optional explainability.

Usage:
    # Basic prediction
    python scripts/predict.py --text "Breaking news: ..." --image photo.jpg

    # With explanations
    python scripts/predict.py --text "..." --image photo.jpg --explain

    # Text-only
    python scripts/predict.py --text "Some suspicious headline"
"""

import os
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import MultimodalPredictor


def main():
    parser = argparse.ArgumentParser(description="Predict Fake News")
    parser.add_argument("--text", type=str, required=True, help="Input text to analyze")
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt",
                        help="Model checkpoint path")
    parser.add_argument("--explain", action="store_true", help="Generate explanations")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load predictor
    import torch
    device = torch.device(args.device) if args.device else None
    predictor = MultimodalPredictor.from_checkpoint(args.checkpoint, device=device)

    # Predict
    if args.explain:
        result = predictor.predict_with_explanation(
            text=args.text,
            image=args.image,
            filename_prefix="cli_prediction",
        )
        print(f"\n{'=' * 50}")
        print(f"  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence']:.1%}")
        print(f"  P(Real)    : {result['probabilities']['real']:.4f}")
        print(f"  P(Fake)    : {result['probabilities']['fake']:.4f}")
        print(f"{'=' * 50}")

        if "explanations" in result:
            print(f"\nExplanation files generated:")
            for key, data in result["explanations"].items():
                if isinstance(data, dict):
                    for subkey, value in data.items():
                        if isinstance(value, str) and os.path.exists(value):
                            print(f"  - {subkey}: {value}")
    else:
        result = predictor.predict(
            text=args.text,
            image=args.image,
            threshold=args.threshold,
        )
        print(f"\n{'=' * 50}")
        print(f"  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence']:.1%}")
        print(f"  P(Real)    : {result['probabilities']['real']:.4f}")
        print(f"  P(Fake)    : {result['probabilities']['fake']:.4f}")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
