# 🔍 Multimodal Transformer Framework for Fake News Detection

A transformer-based multimodal framework combining **Vision Transformer (ViT)** and **BERT** with **cross-modal attention fusion** for robust fake news detection in social media.

> **Conference-Ready**: This framework includes k-fold cross-validation, statistical significance testing (McNemar's, bootstrap CI, paired t-test), and publication-ready LaTeX table generation.

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Key Contributions](#-key-contributions)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Datasets](#-datasets)
- [Training](#-training)
- [K-Fold Cross-Validation](#-k-fold-cross-validation)
- [Evaluation & Ablation](#-evaluation--ablation)
- [Statistical Significance](#-statistical-significance)
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
| 6 | **Statistical Rigor** | K-fold CV, bootstrap CI, McNemar's test, paired t-test, mean±std reporting |
| 7 | **Publication Pipeline** | LaTeX table generation, 300 DPI plots, radar charts, normalized confusion matrices |

---

## 📁 Project Structure

```
multimodal-fake-news-detection/
├── config/
│   └── config.yaml                  # Full configuration (training, CV, stats)
├── src/
│   ├── models/
│   │   ├── text_encoder.py          # BERT-based text encoder
│   │   ├── image_encoder.py         # ViT-based image encoder
│   │   ├── cross_modal_attention.py # Cross-modal attention + gated fusion
│   │   └── multimodal_detector.py   # End-to-end model
│   ├── data/
│   │   ├── dataset.py               # Multi-dataset loaders with adapters
│   │   └── preprocessing.py         # Text cleaning + image augmentation
│   ├── training/
│   │   ├── trainer.py               # Training loop (FP16, warmup, focal loss)
│   │   └── metrics.py               # Metrics + stats + LaTeX tables
│   ├── explainability/
│   │   ├── grad_cam.py              # Grad-CAM for ViT (attention rollout)
│   │   └── attention_viz.py         # Text attention heatmaps + HTML export
│   └── inference/
│       └── predictor.py             # High-level prediction API
├── app/
│   ├── web_app.py                   # Flask REST API server
│   └── streamlit_app.py             # Streamlit demo application
├── scripts/
│   ├── train.py                     # Standard training entry point
│   ├── train_kfold.py               # K-fold cross-validation training
│   ├── evaluate.py                  # Evaluation + ablation + stats
│   ├── predict.py                   # CLI prediction tool
│   └── prepare_isot_dataset.py      # ISOT dataset preparation
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

### Train On Two Datasets Together
```bash
python scripts/train.py \
  --datasets generic generic \
  --data_dirs "ISOT Fake News Dataset" "Fake News Dataset"
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
- Captures **environment snapshot** for reproducibility
- Supports **label smoothing** and **focal loss** for regularization

---

## 🔄 K-Fold Cross-Validation

For conference-level statistical rigor:

### Standard 5-Fold CV
```bash
python scripts/train_kfold.py --config config/config.yaml
```

### Custom Folds and Runs
```bash
python scripts/train_kfold.py \
  --config config/config.yaml \
  --n_folds 10 \
  --n_runs 5
```

### Full Ablation with K-Fold CV
```bash
python scripts/train_kfold.py \
  --config config/config.yaml \
  --ablation
```

This produces:
- **Mean ± std** results across all folds and runs
- **Paired t-tests** between multimodal and unimodal variants
- **LaTeX tables** ready for paper inclusion
- **Radar charts** for visual model comparison

Example output:
```
  K-FOLD CROSS-VALIDATION SUMMARY (multimodal)
  5-Fold CV × 3 Runs = 15 total evaluations
════════════════════════════════════════════════════════════════
  accuracy            : 0.9234 ± 0.0087
  precision           : 0.9198 ± 0.0112
  recall              : 0.9234 ± 0.0087
  f1                  : 0.9215 ± 0.0095
  mcc                 : 0.8468 ± 0.0174
  cohens_kappa        : 0.8467 ± 0.0174
  auc_roc             : 0.9641 ± 0.0062
════════════════════════════════════════════════════════════════
```

---

## 📈 Evaluation & Ablation

### Full Ablation with Statistical Tests
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --ablation \
  --bootstrap \
  --latex
```

This produces:
```
Mode                 Accuracy     Precision    Recall       F1           MCC          AUC-ROC
──────────────────────────────────────────────────────────────────────────────────────────────
multimodal           0.9234       0.9198       0.9234       0.9215       0.8468       0.9641
text_only            0.8756       0.8721       0.8756       0.8738       0.7512       0.9312
image_only           0.7823       0.7801       0.7823       0.7811       0.5646       0.8534
```

### With Bootstrap Confidence Intervals
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --bootstrap
```

### Multi-Dataset Benchmark
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --multi_dataset weibo gossipcop politifact
```

### Evaluate A Combined Two-Dataset Setup
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --datasets generic generic \
  --data_dirs "ISOT Fake News Dataset" "Fake News Dataset"
```

### Generated Outputs
- `results/confusion_matrix.png` (+ `.pdf`) — Confusion matrix
- `results/confusion_matrix_norm.png` — Normalized confusion matrix
- `results/roc_curve.png` (+ `.pdf`) — ROC curve with AUC
- `results/pr_curve.png` (+ `.pdf`) — Precision-Recall curve with iso-F1
- `results/model_comparison.png` (+ `.pdf`) — Ablation bar chart
- `results/ablation_radar.png` (+ `.pdf`) — Radar chart comparison
- `results/ablation_table.tex` — LaTeX-formatted results table

---

## 📐 Statistical Significance

The framework supports three standard tests used in NLP/CV conference papers:

| Test | Use Case | Comparison Type |
|------|----------|----------------|
| **McNemar's Test** | Compare two models on same test set | Paired, per-sample |
| **Bootstrap CI** | Confidence intervals for single model | Single model |
| **Paired t-Test** | Compare models across k-fold runs | Paired, across folds |

### Metrics Reported

| Metric | Symbol | Description |
|--------|--------|-------------|
| Accuracy | Acc | Overall correctness |
| Precision | Prec | Positive predictive value |
| Recall | Rec | True positive rate |
| F1-Score | F1 | Harmonic mean of Prec & Rec |
| MCC | φ | Matthews Correlation Coefficient |
| Cohen's κ | κ | Agreement beyond chance |
| AUC-ROC | AUC | Area under ROC curve |
| Avg Precision | AP | Area under PR curve |

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

---

## ⚙️ Configuration

All settings are in `config/config.yaml`:

| Section | Key Settings |
|---------|-------------|
| **model.text_encoder** | BERT model, max length, frozen layers |
| **model.image_encoder** | ViT model, image size, frozen layers |
| **model.fusion** | Hidden size, attention heads, cross-attention layers |
| **training** | Batch size, LR, epochs, scheduler, label smoothing, focal loss |
| **cross_validation** | K-folds, number of runs, stratified splitting |
| **evaluation** | Metrics list, statistical tests, bootstrap settings |
| **data** | Dataset name, augmentation settings |
| **explainability** | Grad-CAM target layer, attention head aggregation |
| **publication** | LaTeX tables, figure DPI, PDF output |

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
