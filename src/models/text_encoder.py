"""
BERT-based Text Encoder for Fake News Detection
=================================================
Extracts contextual text representations using a pretrained BERT model.
Supports partial layer freezing for efficient fine-tuning.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertTextEncoder(nn.Module):
    """
    BERT-based encoder that extracts token-level and sequence-level
    representations from input text.

    Args:
        model_name (str): HuggingFace model identifier.
        hidden_size (int): BERT hidden dimension (typically 768).
        max_length (int): Maximum token sequence length.
        freeze_layers (int): Number of initial encoder layers to freeze.
        projection_dim (int): Output projection dimension for fusion compatibility.
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        hidden_size: int = 768,
        max_length: int = 256,
        freeze_layers: int = 6,
        projection_dim: int = 512,
    ):
        super().__init__()
        self.max_length = max_length
        self.hidden_size = hidden_size

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name, output_attentions=True)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # Freeze early layers for efficient fine-tuning
        self._freeze_layers(freeze_layers)

        # Project BERT hidden size to fusion dimension
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _freeze_layers(self, num_layers: int):
        """Freeze the embedding layer and the first `num_layers` encoder layers."""
        # Freeze embeddings
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        # Freeze specified encoder layers
        for i in range(min(num_layers, len(self.bert.encoder.layer))):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass through BERT encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]

        Returns:
            dict with keys:
                - 'sequence_output': Projected token-level features [B, seq_len, proj_dim]
                - 'pooled_output': Projected [CLS] representation [B, proj_dim]
                - 'attentions': Tuple of attention weights from each layer
                - 'attention_mask': The input attention mask (for downstream use)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # Token-level representations
        sequence_output = outputs.last_hidden_state          # [B, seq_len, 768]
        projected_sequence = self.projection(sequence_output) # [B, seq_len, proj_dim]

        # [CLS] token representation
        cls_output = sequence_output[:, 0, :]                 # [B, 768]
        projected_cls = self.projection(cls_output)           # [B, proj_dim]

        return {
            "sequence_output": projected_sequence,
            "pooled_output": projected_cls,
            "attentions": outputs.attentions,
            "attention_mask": attention_mask,
        }

    def tokenize(self, texts: list, device: torch.device = None) -> dict:
        """
        Tokenize a batch of raw text strings.

        Args:
            texts: List of input strings.
            device: Target device for tensors.

        Returns:
            Dictionary of tokenized tensors.
        """
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded
