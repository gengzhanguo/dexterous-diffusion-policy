"""
Lightweight image encoder for visual diffusion policy.

Architecture options:
  - ResNet-18  (default, 11M params, ImageNet pretrained)
  - ResNet-50  (larger, 25M params)
  - SmallCNN   (custom tiny CNN, ~200K params, trains from scratch)

The encoder maps (B, 3, H, W) → (B, emb_dim).
Backbone weights can be frozen (recommended for small datasets).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T


# ─────────────────────────────────────────────────────────────────────────── #
# Small custom CNN (no pretrained weights, faster for tiny datasets)           #
# ─────────────────────────────────────────────────────────────────────────── #

class SmallCNN(nn.Module):
    """
    4-layer CNN for 84×84 or 64×64 images.
    Output: (B, emb_dim).
    """
    def __init__(self, in_channels: int = 3, emb_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),  # /4
            nn.Conv2d(32, 64, 4, stride=2),          nn.ReLU(),  # /8
            nn.Conv2d(64, 64, 3, stride=1),          nn.ReLU(),  # /8
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Dropout(dropout),                                  # regularization
            nn.Linear(64 * 4 * 4, emb_dim),
            nn.LayerNorm(emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────── #
# ResNet-based encoder                                                          #
# ─────────────────────────────────────────────────────────────────────────── #

class ResNetEncoder(nn.Module):
    """
    ImageNet-pretrained ResNet backbone + linear projection.

    Args:
        model_name : "resnet18" | "resnet50"
        emb_dim    : output embedding dimension
        frozen     : if True, backbone weights are frozen (fine-tunes only proj)
        pretrained : use ImageNet weights
    """

    _FEAT_DIM = {"resnet18": 512, "resnet50": 2048}

    def __init__(
        self,
        model_name: str = "resnet18",
        emb_dim: int = 256,
        frozen: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.frozen = frozen
        self.emb_dim = emb_dim

        # Load backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = getattr(tvm, model_name)(weights=weights)

        # Strip the FC head → output is (B, feat_dim, 1, 1) after avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        feat_dim = self._FEAT_DIM.get(model_name, 512)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.LayerNorm(emb_dim),
        )

        if frozen:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ImageNet normalization (applied inside forward)
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32 in [0, 1]
        Returns:
            emb: (B, emb_dim)
        """
        # Normalize
        x = (x - self.img_mean) / (self.img_std + 1e-8)

        if self.frozen:
            with torch.no_grad():
                feat = self.backbone(x).flatten(1)   # (B, feat_dim)
        else:
            feat = self.backbone(x).flatten(1)

        return self.proj(feat)                        # (B, emb_dim)


# ─────────────────────────────────────────────────────────────────────────── #
# Multi-camera encoder (stack multiple views)                                   #
# ─────────────────────────────────────────────────────────────────────────── #

class MultiViewEncoder(nn.Module):
    """
    Encodes multiple camera views independently and concatenates embeddings.

    Args:
        num_cameras : number of camera views
        per_cam_dim : embedding dim per camera
        encoder_cls : ResNetEncoder or SmallCNN
    """

    def __init__(
        self,
        num_cameras: int = 2,
        per_cam_dim: int = 128,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoders = nn.ModuleList([
            ResNetEncoder(emb_dim=per_cam_dim, **encoder_kwargs)
            for _ in range(num_cameras)
        ])
        self.emb_dim = num_cameras * per_cam_dim

    def forward(self, images: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: list of (B, 3, H, W) tensors, one per camera
        Returns:
            emb: (B, num_cameras * per_cam_dim)
        """
        embs = [enc(img) for enc, img in zip(self.encoders, images)]
        return torch.cat(embs, dim=-1)


# ─────────────────────────────────────────────────────────────────────────── #
# Factory                                                                        #
# ─────────────────────────────────────────────────────────────────────────── #

def build_image_encoder(
    encoder_type: str = "resnet18",
    emb_dim: int = 256,
    frozen: bool = True,
    pretrained: bool = True,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Factory for image encoders.
    encoder_type: "resnet18" | "resnet50" | "small_cnn"
    """
    if encoder_type == "small_cnn":
        return SmallCNN(emb_dim=emb_dim, dropout=dropout)
    elif encoder_type in ("resnet18", "resnet50"):
        return ResNetEncoder(
            model_name=encoder_type,
            emb_dim=emb_dim,
            frozen=frozen,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
