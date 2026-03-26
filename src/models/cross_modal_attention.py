"""
Cross-Modal Attention Fusion Module
=====================================
Implements bidirectional cross-attention between text and image modalities.
This is the KEY CONTRIBUTION over simple concatenation: it allows each modality
to attend to and enrich itself with information from the other modality.

Architecture:
    Text → attends to Image patches  (Text-guided Visual Attention)
    Image → attends to Text tokens   (Image-guided Textual Attention)
    → Gated fusion of enriched representations
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionLayer(nn.Module):
    """
    Single layer of bidirectional cross-modal attention.

    Given query from one modality and key/value from another, computes
    cross-attention to produce modality-enriched representations.
    """

    def __init__(self, hidden_size: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Text-to-Image cross-attention (text queries, image keys/values)
        self.text_to_image_query = nn.Linear(hidden_size, hidden_size)
        self.text_to_image_key = nn.Linear(hidden_size, hidden_size)
        self.text_to_image_value = nn.Linear(hidden_size, hidden_size)
        self.text_to_image_out = nn.Linear(hidden_size, hidden_size)

        # Image-to-Text cross-attention (image queries, text keys/values)
        self.image_to_text_query = nn.Linear(hidden_size, hidden_size)
        self.image_to_text_key = nn.Linear(hidden_size, hidden_size)
        self.image_to_text_value = nn.Linear(hidden_size, hidden_size)
        self.image_to_text_out = nn.Linear(hidden_size, hidden_size)

        # Layer norms and dropout
        self.text_norm1 = nn.LayerNorm(hidden_size)
        self.text_norm2 = nn.LayerNorm(hidden_size)
        self.image_norm1 = nn.LayerNorm(hidden_size)
        self.image_norm2 = nn.LayerNorm(hidden_size)

        # Feed-forward networks for each modality after cross-attention
        self.text_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )
        self.image_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

        self.attn_dropout = nn.Dropout(dropout)

    def _multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_proj: nn.Linear,
        k_proj: nn.Linear,
        v_proj: nn.Linear,
        out_proj: nn.Linear,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Compute multi-head cross-attention.

        Returns:
            output: Attended features [B, seq_q, hidden_size]
            attn_weights: Attention weights [B, num_heads, seq_q, seq_k]
        """
        B, seq_q, _ = query.shape
        _, seq_k, _ = key.shape

        # Project and reshape to multi-head format
        Q = q_proj(query).view(B, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = k_proj(key).view(B, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = v_proj(value).view(B, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, seq_q, seq_k]

        if mask is not None:
            # Expand mask for multi-head: [B, 1, 1, seq_k]
            mask = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted aggregation
        context = torch.matmul(attn_weights, V)  # [B, H, seq_q, head_dim]
        context = context.transpose(1, 2).contiguous().view(B, seq_q, -1)
        output = out_proj(context)

        return output, attn_weights

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        text_mask: torch.Tensor = None,
    ) -> dict:
        """
        Bidirectional cross-modal attention.

        Args:
            text_features:  [B, seq_text, hidden_size]
            image_features: [B, seq_image, hidden_size]
            text_mask:      [B, seq_text] attention mask for text

        Returns:
            dict with enriched text_features, image_features, and attention maps
        """
        # --- Text attending to Image ---
        text_cross_attn, t2i_weights = self._multi_head_attention(
            query=text_features,
            key=image_features,
            value=image_features,
            q_proj=self.text_to_image_query,
            k_proj=self.text_to_image_key,
            v_proj=self.text_to_image_value,
            out_proj=self.text_to_image_out,
        )
        text_features = self.text_norm1(text_features + text_cross_attn)
        text_features = self.text_norm2(text_features + self.text_ffn(text_features))

        # --- Image attending to Text ---
        image_cross_attn, i2t_weights = self._multi_head_attention(
            query=image_features,
            key=text_features,
            value=text_features,
            q_proj=self.image_to_text_query,
            k_proj=self.image_to_text_key,
            v_proj=self.image_to_text_value,
            out_proj=self.image_to_text_out,
            mask=text_mask,
        )
        image_features = self.image_norm1(image_features + image_cross_attn)
        image_features = self.image_norm2(image_features + self.image_ffn(image_features))

        return {
            "text_features": text_features,
            "image_features": image_features,
            "text_to_image_attention": t2i_weights,
            "image_to_text_attention": i2t_weights,
        }


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism that learns to weight the relative importance
    of text vs. image modalities dynamically per sample.
    """

    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.gate_text = nn.Linear(hidden_size * 2, hidden_size)
        self.gate_image = nn.Linear(hidden_size * 2, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_pooled: torch.Tensor,
        image_pooled: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_pooled:  [B, hidden_size]
            image_pooled: [B, hidden_size]

        Returns:
            fused: [B, hidden_size]
        """
        concat = torch.cat([text_pooled, image_pooled], dim=-1)  # [B, 2*hidden]

        # Compute gating weights
        gate_t = torch.sigmoid(self.gate_text(concat))    # [B, hidden]
        gate_i = torch.sigmoid(self.gate_image(concat))   # [B, hidden]

        # Weighted combination
        fused = gate_t * text_pooled + gate_i * image_pooled
        fused = self.layer_norm(fused)

        return fused


class CrossModalAttentionFusion(nn.Module):
    """
    Full cross-modal fusion module: stacks multiple CrossModalAttentionLayers
    and combines the enriched representations using gated fusion.

    This is the core architectural contribution that replaces simple
    concatenation with learned cross-modal interactions.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cross_attention_layers = nn.ModuleList([
            CrossModalAttentionLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.gated_fusion = GatedFusion(hidden_size)

        # Attention pooling for sequence-to-vector
        self.text_attn_pool = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        self.image_attn_pool = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )

    def _attention_pool(
        self,
        features: torch.Tensor,
        pool_layer: nn.Module,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Attention-weighted pooling from sequence to single vector.

        Args:
            features: [B, seq_len, hidden_size]
            pool_layer: Linear layer producing attention logits
            mask: [B, seq_len] optional mask

        Returns:
            pooled: [B, hidden_size]
        """
        attn_logits = pool_layer(features).squeeze(-1)  # [B, seq_len]
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=-1)   # [B, seq_len]
        pooled = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)  # [B, hidden]
        return pooled

    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        text_mask: torch.Tensor = None,
    ) -> dict:
        """
        Full cross-modal fusion forward pass.

        Args:
            text_features:   [B, seq_text, hidden_size]
            image_features:  [B, seq_image, hidden_size]
            text_mask:       [B, seq_text]

        Returns:
            dict with:
                - 'fused': Final fused representation [B, hidden_size]
                - 'text_enriched': Cross-attention enriched text [B, seq_text, hidden]
                - 'image_enriched': Cross-attention enriched image [B, seq_image, hidden]
                - 'cross_attention_maps': List of attention weight dicts per layer
        """
        all_attention_maps = []

        # Stack cross-attention layers
        for layer in self.cross_attention_layers:
            layer_output = layer(text_features, image_features, text_mask)
            text_features = layer_output["text_features"]
            image_features = layer_output["image_features"]
            all_attention_maps.append({
                "text_to_image": layer_output["text_to_image_attention"],
                "image_to_text": layer_output["image_to_text_attention"],
            })

        # Pool enriched sequences to vectors
        text_pooled = self._attention_pool(text_features, self.text_attn_pool, text_mask)
        image_pooled = self._attention_pool(image_features, self.image_attn_pool)

        # Gated fusion
        fused = self.gated_fusion(text_pooled, image_pooled)

        return {
            "fused": fused,
            "text_enriched": text_features,
            "image_enriched": image_features,
            "text_pooled": text_pooled,
            "image_pooled": image_pooled,
            "cross_attention_maps": all_attention_maps,
        }
