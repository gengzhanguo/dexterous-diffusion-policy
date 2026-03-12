"""
CLIP Text Encoder for language conditioning.

Wraps a frozen OpenCLIP TextTransformer, projects its features to a smaller
`text_emb_dim`, and optionally applies LayerNorm.

Input: raw text string (e.g., "pick up the red cube")
Output: (B, text_emb_dim) float32 embedding
"""
from __future__ import annotations

import torch
import torch.nn as nn

# OpenCLIP (requires pip install open-clip-torch)
import open_clip


class CLIPTextEncoder(nn.Module):
    """
    Frozen OpenCLIP Text Encoder + linear projection.
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B-32",  # e.g., "ViT-B-32", "RN50"
        pretrained: str = "openai",         # "openai", "laion2b_s34b_b79k" etc.
        text_emb_dim: int = 128,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.text_emb_dim = text_emb_dim

        # Load OpenCLIP model
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )

        # Freeze CLIP text encoder
        if freeze_clip:
            for p in self.clip_model.parameters():
                p.requires_grad_(False)
            self.clip_model.eval()

        # Linear projection to desired embedding dimension
        clip_output_dim = self.clip_model.text_projection.shape[1] if hasattr(self.clip_model, 'text_projection') else self.clip_model.embed_dim

        self.proj = nn.Sequential(
            nn.Linear(clip_output_dim, text_emb_dim),
            nn.LayerNorm(text_emb_dim),
        )

        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of strings (B,)
        Returns:
            emb: (B, text_emb_dim)
        """
        device = next(self.parameters()).device
        tokens = self.tokenizer(texts).to(device)

        with torch.no_grad():
            text_features = self.clip_model.encode_text(tokens).float()

        return self.proj(text_features)
