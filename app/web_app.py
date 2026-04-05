"""
Real-Time Prediction Web Application
=======================================
Flask-based web API for real-time fake news detection with
multimodal analysis and explainability.

Endpoints:
    POST /predict          → Basic prediction (text + optional image)
    POST /predict/explain  → Prediction with Grad-CAM + attention explanations
    POST /predict/batch    → Batch prediction
    GET  /health           → Health check
    GET  /model/info       → Model configuration and parameters info
"""

import os
import sys
import json
import base64
import io
import time
import yaml
import torch
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.predictor import MultimodalPredictor


def create_app(config_path: str = None, checkpoint_path: str = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_path: Path to config.yaml
        checkpoint_path: Path to model checkpoint

    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    CORS(app)

    # Load configuration
    if config_path is None:
        config_path = os.path.join(project_root, "config", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load model
    if checkpoint_path is None:
        checkpoint_path = config.get("inference", {}).get(
            "model_checkpoint", "./combined_model_artifacts/checkpoints/best_model_combined.pt"
        )

    predictor = None
    model_loaded = False

    if os.path.exists(checkpoint_path):
        try:
            predictor = MultimodalPredictor.from_checkpoint(checkpoint_path)
            model_loaded = True
            print(f"[APP] Model loaded successfully from {checkpoint_path}")
        except Exception as e:
            print(f"[APP] Failed to load model: {e}")
            print("[APP] Running in demo mode (no predictions available)")
    else:
        print(f"[APP] Checkpoint not found: {checkpoint_path}")
        print("[APP] Running in demo mode. Train a model first.")

    # =========================================================================
    # API Endpoints
    # =========================================================================

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "model_loaded": model_loaded,
            "device": str(predictor.device) if predictor else "none",
            "timestamp": time.time(),
        })

    @app.route("/model/info", methods=["GET"])
    def model_info():
        """Return model configuration and parameter info."""
        info = {
            "model_loaded": model_loaded,
            "config": {
                "text_encoder": config["model"]["text_encoder"]["name"],
                "image_encoder": config["model"]["image_encoder"]["name"],
                "fusion_hidden_size": config["model"]["fusion"]["hidden_size"],
                "num_cross_attn_layers": config["model"]["fusion"]["num_cross_attn_layers"],
                "num_classes": config["model"]["classifier"]["num_classes"],
            },
        }
        if predictor and model_loaded:
            params = predictor.model.get_trainable_params()
            info["parameters"] = params
        return jsonify(info)

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Basic prediction endpoint.

        Accepts JSON:
            {
                "text": "news headline or article text",
                "image": "base64-encoded image string (optional)",
                "image_url": "URL to image (optional)",
                "threshold": 0.5 (optional)
            }

        Or multipart form data with 'text' and 'image' fields.
        """
        if not model_loaded:
            return jsonify({"error": "Model not loaded. Train a model first."}), 503

        try:
            text, image, threshold = _parse_request(request)

            if not text:
                return jsonify({"error": "No text provided"}), 400

            result = predictor.predict(
                text=text,
                image=image,
                threshold=threshold,
            )

            return jsonify({
                "success": True,
                "result": result,
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/predict/explain", methods=["POST"])
    def predict_with_explanation():
        """
        Prediction with Grad-CAM and text attention explanations.

        Returns prediction plus paths to generated explanation visualizations.
        """
        if not model_loaded:
            return jsonify({"error": "Model not loaded"}), 503

        try:
            text, image, threshold = _parse_request(request)

            if not text:
                return jsonify({"error": "No text provided"}), 400

            # Generate timestamp-based prefix for unique filenames
            prefix = f"sample_{int(time.time())}"

            result = predictor.predict_with_explanation(
                text=text,
                image=image,
                filename_prefix=prefix,
            )

            # Convert explanation image paths to base64 for API response
            explanations_b64 = {}
            if "explanations" in result:
                for key, exp_data in result["explanations"].items():
                    if isinstance(exp_data, dict):
                        for subkey, value in exp_data.items():
                            if isinstance(value, str) and value.endswith(".png"):
                                if os.path.exists(value):
                                    with open(value, "rb") as f:
                                        img_b64 = base64.b64encode(f.read()).decode("utf-8")
                                    explanations_b64[f"{key}_{subkey}"] = img_b64

            response = {
                "success": True,
                "result": {
                    "prediction": result["prediction"],
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                    "probabilities": result["probabilities"],
                },
                "explanations": explanations_b64,
            }

            return jsonify(response)

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    @app.route("/predict/batch", methods=["POST"])
    def predict_batch():
        """
        Batch prediction endpoint.

        Accepts JSON:
            {
                "samples": [
                    {"text": "...", "image": "base64 (optional)"},
                    ...
                ]
            }
        """
        if not model_loaded:
            return jsonify({"error": "Model not loaded"}), 503

        try:
            data = request.get_json()
            samples = data.get("samples", [])

            if not samples:
                return jsonify({"error": "No samples provided"}), 400

            texts = [s.get("text", "") for s in samples]
            images = []
            for s in samples:
                if "image" in s and s["image"]:
                    img_bytes = base64.b64decode(s["image"])
                    images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                else:
                    images.append(None)

            results = predictor.predict_batch(texts=texts, images=images)

            return jsonify({
                "success": True,
                "results": results,
                "count": len(results),
            })

        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


def _parse_request(req) -> tuple:
    """Parse text and image from request (supports JSON and multipart form)."""
    text = None
    image = None
    threshold = 0.5

    if req.is_json:
        data = req.get_json()
        text = data.get("text", "")
        threshold = data.get("threshold", 0.5)

        # Base64 image
        if "image" in data and data["image"]:
            img_bytes = base64.b64decode(data["image"])
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    else:
        # Multipart form data
        text = req.form.get("text", "")
        threshold = float(req.form.get("threshold", 0.5))

        if "image" in req.files:
            img_file = req.files["image"]
            image = Image.open(img_file.stream).convert("RGB")

    return text, image, threshold


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal Fake News Detection API")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(config_path=args.config, checkpoint_path=args.checkpoint)
    app.run(host=args.host, port=args.port, debug=args.debug)
