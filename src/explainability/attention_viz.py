"""
Text Attention Visualization for Explainable Fake News Detection
==================================================================
Visualizes which words/tokens the model attends to when making its
prediction, using both:
    1. BERT self-attention weights
    2. Cross-modal attention (text → image) weights

Produces highlighted text visualizations and HTML-based attention maps.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from transformers import BertTokenizer


class TextAttentionVisualizer:
    """
    Visualizes text attention patterns from the multimodal detector.

    Supports:
        - BERT self-attention visualization (per-layer, per-head)
        - Cross-modal text-to-image attention (which text tokens look at which patches)
        - Token importance scoring
        - HTML export for interactive viewing
    """

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        save_dir: str = "./explanations",
        layer_index: int = -1,
        head_aggregation: str = "mean",
    ):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.layer_index = layer_index
        self.head_aggregation = head_aggregation

    def extract_self_attention(
        self,
        attentions: tuple,
        input_ids: np.ndarray,
        attention_mask: np.ndarray = None,
    ) -> dict:
        """
        Extract and process BERT self-attention weights.

        Args:
            attentions: Tuple of attention tensors from BERT [num_layers x (B, H, seq, seq)]
            input_ids: Token IDs [seq_len]
            attention_mask: Attention mask [seq_len]

        Returns:
            dict with tokens, attention matrix, and token importance scores
        """
        # Select target layer
        attn = attentions[self.layer_index]  # [B, H, seq, seq]

        if hasattr(attn, "detach"):
            attn = attn.detach().cpu().numpy()

        # Take first sample in batch
        if attn.ndim == 4:
            attn = attn[0]  # [H, seq, seq]

        # Aggregate heads
        if self.head_aggregation == "mean":
            attn_agg = attn.mean(axis=0)  # [seq, seq]
        elif self.head_aggregation == "max":
            attn_agg = attn.max(axis=0)   # [seq, seq]
        else:
            attn_agg = attn.mean(axis=0)

        # Decode tokens
        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        if input_ids.ndim > 1:
            input_ids = input_ids[0]

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Determine actual sequence length (exclude padding)
        if attention_mask is not None:
            if hasattr(attention_mask, "numpy"):
                attention_mask = attention_mask.numpy()
            if attention_mask.ndim > 1:
                attention_mask = attention_mask[0]
            seq_len = int(attention_mask.sum())
        else:
            seq_len = len([t for t in tokens if t != "[PAD]"])

        # Trim to actual length
        tokens = tokens[:seq_len]
        attn_agg = attn_agg[:seq_len, :seq_len]

        # Compute token importance: [CLS] attention to each token
        cls_attention = attn_agg[0, :]  # [seq_len]
        token_importance = cls_attention / (cls_attention.sum() + 1e-8)

        # Also compute average attention received by each token
        avg_received = attn_agg.mean(axis=0)  # [seq_len]

        return {
            "tokens": tokens,
            "attention_matrix": attn_agg,
            "cls_attention": cls_attention,
            "token_importance": token_importance,
            "avg_attention_received": avg_received,
            "all_heads": attn[:, :seq_len, :seq_len],
        }

    def extract_cross_modal_attention(
        self,
        cross_attention_maps: list,
        input_ids: np.ndarray,
        attention_mask: np.ndarray = None,
    ) -> dict:
        """
        Extract cross-modal attention weights (text-to-image and image-to-text).

        Args:
            cross_attention_maps: List of dicts with 'text_to_image' and 'image_to_text' keys
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            dict with cross-modal attention information
        """
        if not cross_attention_maps:
            return {}

        # Use the last cross-attention layer
        last_layer = cross_attention_maps[-1]

        # Text → Image attention: [B, H, seq_text, seq_image]
        t2i = last_layer["text_to_image"]
        if hasattr(t2i, "detach"):
            t2i = t2i.detach().cpu().numpy()
        if t2i.ndim == 4:
            t2i = t2i[0]  # [H, seq_text, seq_image]

        # Aggregate heads
        t2i_agg = t2i.mean(axis=0)  # [seq_text, seq_image]

        # Image → Text attention: [B, H, seq_image, seq_text]
        i2t = last_layer["image_to_text"]
        if hasattr(i2t, "detach"):
            i2t = i2t.detach().cpu().numpy()
        if i2t.ndim == 4:
            i2t = i2t[0]  # [H, seq_image, seq_text]
        i2t_agg = i2t.mean(axis=0)  # [seq_image, seq_text]

        # Decode tokens
        if hasattr(input_ids, "numpy"):
            input_ids = input_ids.numpy()
        if input_ids.ndim > 1:
            input_ids = input_ids[0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Trim
        if attention_mask is not None:
            if hasattr(attention_mask, "numpy"):
                attention_mask = attention_mask.numpy()
            if attention_mask.ndim > 1:
                attention_mask = attention_mask[0]
            seq_len = int(attention_mask.sum())
        else:
            seq_len = len([t for t in tokens if t != "[PAD]"])

        tokens = tokens[:seq_len]
        t2i_agg = t2i_agg[:seq_len, :]
        i2t_agg = i2t_agg[:, :seq_len]

        # Per-token average attention to image (how much each token looks at the image)
        text_to_image_importance = t2i_agg.mean(axis=1)  # [seq_text]
        # Per-patch average attention to text (how much each patch looks at text)
        image_to_text_importance = i2t_agg.mean(axis=1)  # [seq_image]

        return {
            "tokens": tokens,
            "text_to_image_attention": t2i_agg,
            "image_to_text_attention": i2t_agg,
            "text_to_image_importance": text_to_image_importance,
            "image_to_text_importance": image_to_text_importance,
        }

    def plot_token_importance(
        self,
        tokens: list,
        importance: np.ndarray,
        title: str = "Token Importance",
        filename: str = "token_importance.png",
        top_k: int = 30,
    ) -> str:
        """
        Bar chart of top-K most important tokens.
        """
        # Filter out special tokens
        filtered = [
            (t, s) for t, s in zip(tokens, importance)
            if t not in ["[CLS]", "[SEP]", "[PAD]"]
        ]
        if not filtered:
            return ""

        filtered.sort(key=lambda x: x[1], reverse=True)
        filtered = filtered[:top_k]
        tokens_plot, scores = zip(*filtered)

        fig, ax = plt.subplots(figsize=(12, max(6, len(tokens_plot) * 0.35)))

        colors = plt.cm.YlOrRd(np.array(scores) / max(scores))
        bars = ax.barh(range(len(tokens_plot)), scores, color=colors)

        ax.set_yticks(range(len(tokens_plot)))
        ax.set_yticklabels(tokens_plot, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Attention Score", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[ATTENTION] Token importance plot saved to {save_path}")
        return str(save_path)

    def plot_attention_heatmap(
        self,
        tokens: list,
        attention_matrix: np.ndarray,
        title: str = "Self-Attention Heatmap",
        filename: str = "attention_heatmap.png",
        max_tokens: int = 40,
    ) -> str:
        """
        Plot attention matrix as a heatmap.
        """
        n = min(len(tokens), max_tokens)
        tokens_trunc = tokens[:n]
        attn_trunc = attention_matrix[:n, :n]

        fig, ax = plt.subplots(figsize=(max(10, n * 0.4), max(8, n * 0.35)))

        im = ax.imshow(attn_trunc, cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(tokens_trunc, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens_trunc, fontsize=8)
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()

        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"[ATTENTION] Heatmap saved to {save_path}")
        return str(save_path)

    def generate_highlighted_html(
        self,
        tokens: list,
        importance: np.ndarray,
        prediction: int,
        confidence: float,
        filename: str = "text_explanation.html",
    ) -> str:
        """
        Generate an HTML file with color-highlighted tokens based on importance.
        """
        class_names = ["Real", "Fake"]
        pred_label = class_names[prediction]
        pred_color = "#E91E63" if prediction == 1 else "#4CAF50"

        # Normalize importance to [0, 1]
        if importance.max() > 0:
            norm_importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        else:
            norm_importance = importance

        html_parts = []
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Text Attention Explanation</title>
    <style>
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
            background: #1a1a2e;
            color: #e0e0e0;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: #16213e;
            border-radius: 12px;
            border: 1px solid #0f3460;
        }}
        .prediction {{
            font-size: 28px;
            font-weight: bold;
            color: {pred_color};
        }}
        .confidence {{
            font-size: 18px;
            color: #aaa;
            margin-top: 8px;
        }}
        .text-container {{
            line-height: 2.2;
            font-size: 16px;
            padding: 25px;
            background: #16213e;
            border-radius: 12px;
            border: 1px solid #0f3460;
        }}
        .token {{
            padding: 3px 6px;
            margin: 2px;
            border-radius: 4px;
            display: inline-block;
            transition: transform 0.2s;
        }}
        .token:hover {{
            transform: scale(1.1);
            cursor: pointer;
        }}
        .legend {{
            margin-top: 20px;
            text-align: center;
            padding: 15px;
        }}
        .legend-gradient {{
            height: 20px;
            background: linear-gradient(to right, rgba(255,235,59,0.2), rgba(255,87,34,1));
            border-radius: 4px;
            margin: 10px auto;
            max-width: 300px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="prediction">Prediction: {pred_label}</div>
        <div class="confidence">Confidence: {confidence:.1%}</div>
    </div>
    <div class="text-container">
""")

        for token, score in zip(tokens, norm_importance):
            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            # Clean subword tokens
            display_token = token.replace("##", "")

            # Color: low importance = yellow/transparent, high = red/opaque
            r = int(255)
            g = int(235 - score * 185)
            b = int(59 - score * 59)
            alpha = 0.15 + score * 0.85

            html_parts.append(
                f'<span class="token" style="background-color: rgba({r},{g},{b},{alpha:.2f})" '
                f'title="Attention: {score:.4f}">{display_token}</span> '
            )

        html_parts.append("""
    </div>
    <div class="legend">
        <p>Token Attention Intensity</p>
        <div class="legend-gradient"></div>
        <p style="font-size: 12px; color: #888;">Low ← Attention Score → High</p>
    </div>
</body>
</html>
""")

        save_path = self.save_dir / filename
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("".join(html_parts))

        print(f"[ATTENTION] HTML explanation saved to {save_path}")
        return str(save_path)

    def explain(
        self,
        model_outputs: dict,
        input_ids,
        attention_mask=None,
        filename_prefix: str = "text",
    ) -> dict:
        """
        End-to-end text explanation pipeline.

        Generates token importance bar chart, attention heatmap,
        and highlighted HTML, from model output.

        Args:
            model_outputs: Output dict from MultimodalFakeNewsDetector
            input_ids: Token IDs
            attention_mask: Attention mask
            filename_prefix: Prefix for saved files

        Returns:
            dict with all generated file paths and attention data
        """
        result = {}

        # --- BERT Self-Attention ---
        if "text_attentions" in model_outputs and model_outputs["text_attentions"] is not None:
            self_attn = self.extract_self_attention(
                model_outputs["text_attentions"], input_ids, attention_mask
            )
            result["self_attention"] = self_attn

            # Plot token importance
            result["token_importance_path"] = self.plot_token_importance(
                self_attn["tokens"],
                self_attn["token_importance"],
                title="BERT Token Importance ([CLS] Attention)",
                filename=f"{filename_prefix}_token_importance.png",
            )

            # Plot attention heatmap
            result["attention_heatmap_path"] = self.plot_attention_heatmap(
                self_attn["tokens"],
                self_attn["attention_matrix"],
                title="BERT Self-Attention (Last Layer)",
                filename=f"{filename_prefix}_attention_heatmap.png",
            )

            # Generate HTML
            prediction = model_outputs["logits"].argmax(dim=-1).item() if hasattr(model_outputs["logits"], "argmax") else 0
            confidence = model_outputs["probabilities"].max().item() if hasattr(model_outputs["probabilities"], "max") else 0.5

            result["html_path"] = self.generate_highlighted_html(
                self_attn["tokens"],
                self_attn["token_importance"],
                prediction=prediction,
                confidence=confidence,
                filename=f"{filename_prefix}_explanation.html",
            )

        # --- Cross-Modal Attention ---
        if "cross_attention_maps" in model_outputs and model_outputs["cross_attention_maps"]:
            cross_attn = self.extract_cross_modal_attention(
                model_outputs["cross_attention_maps"], input_ids, attention_mask
            )
            result["cross_attention"] = cross_attn

            if "tokens" in cross_attn and "text_to_image_importance" in cross_attn:
                result["cross_modal_importance_path"] = self.plot_token_importance(
                    cross_attn["tokens"],
                    cross_attn["text_to_image_importance"],
                    title="Text→Image Cross-Attention",
                    filename=f"{filename_prefix}_cross_modal_importance.png",
                )

        return result
