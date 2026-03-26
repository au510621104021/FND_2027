# 🔍 Multimodal Transformer Framework for Fake News Detection

A transformer-based multimodal framework combining **Vision Transformer (ViT)** and **BERT** with **cross-modal attention fusion** for robust fake news detection in social media.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Key Contributions](#-key-contributions)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Datasets](#-datasets)
- [Training](#-training)
- [Evaluation & Ablation](#-evaluation--ablation)
- [Inference](#-inference)
- [Explainability](#-explainability)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Citation](#-citation)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INPUT (Social Media Post)                        │
│                  ┌──────────┐  ┌──────────┐                        │
│                  │   Text   │  │  Image   │                        │
│                  └────┬─────┘  └────┬─────┘                        │
│                       │             │                               │
│              ┌────────▼────────┐ ┌──▼────────────┐                 │
│              │  Preprocessing  │ │ Preprocessing  │                 │
│              │  (Clean, Token) │ │ (Resize, Aug)  │                 │
│              └────────┬────────┘ └──┬────────────┘                 │
│                       │             │                               │
│              ┌────────▼────────┐ ┌──▼────────────┐                 │
│              │   BERT Encoder  │ │  ViT Encoder   │                │
│              │  (bert-base)    │ │ (vit-base-p16) │                │
│              │  [B, S, 768]    │ │ [B, P+1, 768]  │                │
│              └────────┬────────┘ └──┬────────────┘                 │
│                       │             │                               │
│              ┌────────▼─────────────▼────────────┐                 │
│              │    Projection Layers (→ 512D)      │                 │
│              └────────┬─────────────┬────────────┘                 │
│                       │             │                               │
│              ┌────────▼─────────────▼────────────┐                 │
│              │   Cross-Modal Attention (×2)       │                 │
│              │   ┌─────────────────────────────┐  │                │
│              │   │ Text → attends to → Image   │  │                │
│              │   │ Image → attends to → Text   │  │                │
│              │   └─────────────────────────────┘  │                │
│              └────────┬─────────────┬────────────┘                 │
│                       │             │                               │
│              ┌────────▼─────────────▼────────────┐                 │
│              │     Gated Fusion + Attn Pooling    │                │
│              │     σ(W·[t;i]) ⊙ t + σ(W·[t;i]) ⊙ i              │
│              └──────────────┬────────────────────┘                 │
│                             │                                      │
│              ┌──────────────▼────────────────────┐                 │
│              │      Classification Head           │                │
│              │      512 → 256 → 128 → 2          │                │
│              └──────────────┬────────────────────┘                 │
│                             │                                      │
│                    ┌────────▼────────┐                              │
│                    │  Real  │  Fake  │                              │
│                    └─────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🌟 Key Contributions

| # | Contribution | Description |
|---|-------------|-------------|
| 1 | **Cross-Modal Attention** | Bidirectional attention replacing simple concatenation — text attends to image patches and vice versa |
| 2 | **Dual Transformer Architecture** | ViT for images + BERT for text, both with partial layer freezing |
| 3 | **Gated Fusion** | Learned gating mechanism dynamically weights modality importance per sample |
| 4 | **Explainable AI** | Grad-CAM for image regions + attention visualization for text tokens |
| 5 | **Multi-Dataset Evaluation** | Unified framework supporting Weibo, Twitter MediaEval, GossipCop, PolitiFact, Fakeddit |
| 6 | **Real-Time API** | Flask-based REST API for deployment with batch prediction support |

---

## 📁 Project Structure

```
multimodal-fake-news-detection/
├── config/
│   └── config.yaml              # Full configuration file
├── src/
│   ├── models/
│   │   ├── text_encoder.py      # BERT-based text encoder
│   │   ├── image_encoder.py     # ViT-based image encoder
│   │   ├── cross_modal_attention.py  # Cross-modal attention + gated fusion
│   │   └── multimodal_detector.py    # End-to-end model
│   ├── data/
│   │   ├── dataset.py           # Multi-dataset loaders with adapters
│   │   └── preprocessing.py     # Text cleaning + image augmentation
│   ├── training/
│   │   ├── trainer.py           # Training loop (FP16, warmup, early stopping)
│   │   └── metrics.py           # Metrics + publication-ready plots
│   ├── explainability/
│   │   ├── grad_cam.py          # Grad-CAM for ViT (attention rollout)
│   │   └── attention_viz.py     # Text attention heatmaps + HTML export
│   └── inference/
│       └── predictor.py         # High-level prediction API
├── app/
│   └── web_app.py               # Flask REST API server
├── scripts/
│   ├── train.py                 # Training entry point
│   ├── evaluate.py              # Evaluation + ablation studies
│   └── predict.py               # CLI prediction tool
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/multimodal-fake-news-detection.git
cd multimodal-fake-news-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with ≥8GB VRAM (recommended)
- **RAM**: ≥16GB
- **Storage**: ≥10GB (for pretrained models and datasets)

---

## 📊 Datasets

The framework supports benchmark datasets through a unified adapter system:

| Dataset | Source | Labels | Type |
|---------|--------|--------|------|
| **Weibo** | Chinese social media | Rumor / Non-rumor | Text + Image |
| **Twitter MediaEval** | Twitter | Fake / Real | Text + Image |
| **GossipCop** | FakeNewsNet | Fake / Real | Article + Image |
| **PolitiFact** | FakeNewsNet | Fake / Real | Article + Image |
| **Fakeddit** | Reddit | Fake / Real | Title + Image |
| **ISOT** | University of Victoria | Fake / Real | Text-only (image optional) |

### Data Preparation

Place your dataset files in the `data/` directory following the expected structure for each dataset (see `src/data/dataset.py` for adapter-specific formats).

For a **custom dataset**, use CSV/TSV files with columns:
- text-like: `text` or `content` or `statement` or `headline` or `title`
- label-like: `label` or `target` or `class` or `fake` or `category`
- optional image: `image_path` or `image` (missing images are handled gracefully)

You can provide:
- a single file (`dataset.csv`, `data.csv`)
- split files (`train.csv` + `test.csv` + optional `val.csv`)

### ISOT Quick Start

1. Download the ISOT dataset (`Fake.csv` and `True.csv`).
2. Place both files in `data/isot/`.
3. Prepare unified CSV:

```bash
python scripts/prepare_isot_dataset.py --data_dir data/isot
```

4. Train using config defaults (`dataset_name: isot`, `data_dir: ./data/isot`):

```bash
python scripts/train.py --config config/config.yaml
```

---

## 🏋️ Training

### Basic Training (Multimodal)
```bash
python scripts/train.py --config config/config.yaml
```

### Ablation: Text-Only
```bash
python scripts/train.py --config config/config.yaml --mode text_only
```

### Ablation: Image-Only
```bash
python scripts/train.py --config config/config.yaml --mode image_only
```

### Override Hyperparameters
```bash
python scripts/train.py \
  --dataset gossipcop \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-5
```

### Resume Training
```bash
python scripts/train.py --resume checkpoints/latest_model.pt
```

Training automatically:
- Logs to **TensorBoard** (`logs/tensorboard/`)
- Saves **best model** checkpoint (`checkpoints/best_model.pt`)
- Applies **early stopping** when validation loss plateaus
- Uses **mixed-precision (FP16)** for memory efficiency

---

## 📈 Evaluation & Ablation

### Full Ablation Study
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --ablation
```

This produces a comparison table:
```
Mode                 Accuracy     Precision    Recall       F1           AUC-ROC
────────────────────────────────────────────────────────────────────────────────
multimodal           0.9234       0.9198       0.9234       0.9215       0.9641
text_only            0.8756       0.8721       0.8756       0.8738       0.9312
image_only           0.7823       0.7801       0.7823       0.7811       0.8534
```

### Multi-Dataset Benchmark
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --multi_dataset weibo gossipcop politifact
```

### Generated Plots
- `results/confusion_matrix.png` - Confusion matrix
- `results/roc_curve.png` - ROC curve with AUC
- `results/pr_curve.png` - Precision-Recall curve
- `results/model_comparison.png` - Ablation comparison chart

---

## 🔮 Inference

### CLI Prediction
```bash
# With image
python scripts/predict.py \
  --text "Breaking: Scientists discover new species" \
  --image photo.jpg

# With explanations
python scripts/predict.py \
  --text "Shocking news everyone must see!!!" \
  --image suspicious.jpg \
  --explain
```

### Python API
```python
from src.inference.predictor import MultimodalPredictor

predictor = MultimodalPredictor.from_checkpoint("checkpoints/best_model.pt")

# Basic prediction
result = predictor.predict(
    text="Breaking news headline",
    image="path/to/image.jpg"
)
print(result)
# {'prediction': 'Fake', 'confidence': 0.92, 'probabilities': {'real': 0.08, 'fake': 0.92}}

# With explanations (Grad-CAM + attention)
result = predictor.predict_with_explanation(
    text="Some news article...",
    image="image.jpg"
)
```

---

## 🧠 Explainability

### Grad-CAM (Image Explanation)
Shows which image regions the model focuses on:

```python
from src.explainability.grad_cam import MultimodalGradCAM

cam = MultimodalGradCAM(model, device)
result = cam.explain(original_image, input_ids, attention_mask, pixel_values)
# Saves: original | heatmap | overlay visualization
```

### Text Attention Visualization
Highlights important tokens with attention scores:

```python
from src.explainability.attention_viz import TextAttentionVisualizer

viz = TextAttentionVisualizer()
result = viz.explain(model_outputs, input_ids, attention_mask)
# Generates: token importance bar chart, attention heatmap, interactive HTML
```

---

## 🌐 API Reference

### Start the Server
```bash
python app/web_app.py --port 5000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Model configuration and parameters |
| `POST` | `/predict` | Basic prediction |
| `POST` | `/predict/explain` | Prediction + explainability |
| `POST` | `/predict/batch` | Batch prediction |

### Example Request
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Scientists find cure for common cold",
    "image": "<base64-encoded-image>",
    "threshold": 0.5
  }'
```

---

## ⚙️ Configuration

All settings are in `config/config.yaml`:

| Section | Key Settings |
|---------|-------------|
| **model.text_encoder** | BERT model, max length, frozen layers |
| **model.image_encoder** | ViT model, image size, frozen layers |
| **model.fusion** | Hidden size, attention heads, cross-attention layers |
| **training** | Batch size, LR, epochs, scheduler, early stopping |
| **data** | Dataset name, augmentation settings |
| **explainability** | Grad-CAM target layer, attention head aggregation |

---

## 📝 Citation

```bibtex
@article{multimodal_fake_news_2026,
  title={Multimodal Transformer Framework for Fake News Detection using Cross-Modal Attention},
  author={},
  year={2026},
  journal={}
}
```

---

## 📄 License

This project is licensed under the MIT License.
