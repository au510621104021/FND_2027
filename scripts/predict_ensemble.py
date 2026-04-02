"""
Prediction using two or more trained checkpoints.

Usage:
    python scripts/predict_ensemble.py --text "headline" --image sample.jpg \
        --checkpoints checkpoints/best_model.pt checkpoints/fake_or_real/best_model.pt
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.ensemble import EnsemblePredictor


def main():
    parser = argparse.ArgumentParser(description="Predict fake news with an ensemble of checkpoints")
    parser.add_argument("--text", type=str, required=True, help="Input text to analyze")
    parser.add_argument("--image", type=str, default=None, help="Path to image")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint paths to ensemble")
    parser.add_argument("--weights", nargs="+", type=float, default=None, help="Optional ensemble weights")
    parser.add_argument("--mode", type=str, default="multimodal",
                        choices=["multimodal", "text_only", "image_only"],
                        help="Inference mode")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    import torch

    device = torch.device(args.device) if args.device else None
    ensemble = EnsemblePredictor.from_checkpoints(
        checkpoints=args.checkpoints,
        device=device,
        weights=args.weights,
    )

    result = ensemble.predict(
        text=args.text,
        image=args.image,
        mode=args.mode,
        threshold=args.threshold,
    )

    print(f"\n{'=' * 60}")
    print(f"  Ensemble Prediction : {result['prediction']}")
    print(f"  Confidence          : {result['confidence']:.1%}")
    print(f"  P(Real)             : {result['probabilities']['real']:.4f}")
    print(f"  P(Fake)             : {result['probabilities']['fake']:.4f}")
    print(f"{'=' * 60}")

    for idx, individual in enumerate(result["individual_results"], start=1):
        print(f"  Model {idx}: {individual['prediction']} "
              f"(real={individual['probabilities']['real']:.4f}, fake={individual['probabilities']['fake']:.4f})")


if __name__ == "__main__":
    main()
