#!/usr/bin/env python3
"""
model.py — All CNN architectures for RF intrusion detection.

Models (from simplest to most complex):
  1. SVM Baseline         — sklearn SVM on handcrafted features (non-neural)
  2. MLP Baseline         — simple MLP on flattened IQ
  3. CNN1D_IQ             — 1D-CNN on raw IQ [W, 2]
  4. CNN2D_Spec           — 2D-CNN on STFT spectrogram [1, F, T]
  5. DualBranchFusionCNN  — Main model: 1D-IQ + 2D-Spec fusion (OUR CONTRIBUTION)
  6. DualBranchLite       — Lightweight variant for edge devices

Each model exposes:
  - forward(x) → logits [B, num_classes]
  - get_features(x) → embedding vector (for t-SNE)
  - count_params() → total trainable parameters
"""

from functools import partial
import math
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tv_models

from src.data_loader import (
    STFT_HOP,
    STFT_N_FFT,
    WINDOW_SIZE as DATA_WINDOW_SIZE,
)


NUM_CLASSES = 4
WINDOW_SIZE = DATA_WINDOW_SIZE
SPEC_FREQ_BINS = STFT_N_FFT
SPEC_TIME_BINS = 1 + WINDOW_SIZE // STFT_HOP


# ============================================================
# Utility Layers
# ============================================================
class ConvBNReLU1D(nn.Module):
    """Conv1D → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvBNReLU2D(nn.Module):
    """Conv2D → BatchNorm → ReLU block."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation block for 1D features."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, L]
        w = self.fc(x).unsqueeze(-1)  # [B, C, 1]
        return x * w


class SEBlock2D(nn.Module):
    """Squeeze-and-Excitation block for 2D features."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, C, H, W]
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * w


# ============================================================
# Advanced Utility Layers for SOTA
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class DepthwiseSeparableConv2D(nn.Module):
    """Depthwise-separable 2D block for a stronger but cheaper spectrogram branch."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConcatResidualFusion(nn.Module):
    """Fuse branch embeddings with concat -> MLP plus a residual projection."""

    def __init__(self, iq_dim: int, spec_dim: int, out_dim: int, dropout: float = 0.15):
        super().__init__()
        fused_dim = iq_dim + spec_dim
        hidden_dim = max(out_dim, fused_dim // 2)
        self.residual = nn.Linear(fused_dim, out_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, iq_feat: torch.Tensor, spec_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([iq_feat, spec_feat], dim=1)
        return self.norm(self.residual(fused) + self.mlp(fused))

class GatedFusion(nn.Module):
    """Gated Attention fusion of time (1D) and frequency (2D) features"""
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 2)  # 2 scores: one for Branch 1, one for Branch 2
        )
        
    def forward(self, f1, f2):
        # f1, f2 shape: [B, dim]
        concat_f = torch.cat([f1, f2], dim=-1) # [B, 2*dim]
        # Softmax over the 2 branches to create a strict gating (confidence) mechanism
        scores = F.softmax(self.attention(concat_f), dim=-1) # [B, 2]
        
        score1 = scores[:, 0].unsqueeze(1) # [B, 1]
        score2 = scores[:, 1].unsqueeze(1) # [B, 1]
        
        # Combine the two domains according to their predicted confidence
        fused = f1 * score1 + f2 * score2
        return fused


# ============================================================
# Model 1: MLP Baseline
# ============================================================
class MLPBaseline(nn.Module):
    """
    Simple MLP on flattened raw IQ.

    Input: [B, W, 2] → flatten → MLP → [B, num_classes]
    """

    def __init__(self, window_size=WINDOW_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        self.flatten_dim = window_size * 2
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.feature_layer = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
        """x: [B, W, 2]"""
        return self.model(x)

    def get_features(self, x):
        """Extract 64-dim feature embedding."""
        return self.feature_layer(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Model 2: 1D-CNN on Raw IQ
# ============================================================
class CNN1D_IQ(nn.Module):
    """
    1D Convolutional Network on raw IQ data.

    Input: [B, W, 2] → permute → [B, 2, W] → 1D-CNN → [B, num_classes]
    """

    def __init__(self, window_size=WINDOW_SIZE, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU1D(2, 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(2),
            ConvBNReLU1D(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(2),
            ConvBNReLU1D(64, 128, kernel_size=3, stride=1, padding=1),
            SEBlock1D(128),
            nn.MaxPool1d(2),
            ConvBNReLU1D(128, 256, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """x: [B, W, 2]"""
        x = x.permute(0, 2, 1)  # [B, 2, W]
        features = self.encoder(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extract 256-dim embedding before classifier."""
        x = x.permute(0, 2, 1)
        features = self.encoder(x)
        return features.flatten(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Model 3: 2D-CNN on Spectrogram
# ============================================================
class CNN2D_Spec(nn.Module):
    """
    2D Convolutional Network on STFT spectrogram.

    Input: [B, 1, F, T] → 2D-CNN → [B, num_classes]
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBNReLU2D(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBNReLU2D(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBNReLU2D(64, 128, kernel_size=3, padding=1),
            SEBlock2D(128),
            nn.MaxPool2d(2),
            ConvBNReLU2D(128, 256, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """x: [B, 1, F, T]"""
        features = self.encoder(x)
        return self.classifier(features)

    def get_features(self, x):
        """Extract 256-dim embedding."""
        features = self.encoder(x)
        return features.flatten(1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Model 4: Dual-Branch Fusion CNN (MAIN MODEL — Our Contribution)
# ============================================================
class DualBranchFusionCNN(nn.Module):
    """
    SOTA Dual-branch CNN with Gated Feature Fusion for RF intrusion detection.
    
    Branch 1: 1D ResNet on raw IQ [B, W, 2]
    Branch 2: 2D ResNet + CBAM Attention on STFT spectrogram [B, 1, F, T]
    Fusion:   Gated Attention Fusion -> MLP -> 4-class softmax
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        num_classes: int = NUM_CLASSES,
        branch1_dim: int = 128,
        branch2_dim: int = 128,
        fusion_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        # For gated attention to work symmetrically, branches must output same dimension size
        if branch1_dim != branch2_dim:
            branch1_dim = max(branch1_dim, branch2_dim)
            branch2_dim = max(branch1_dim, branch2_dim)
            
        dim = branch1_dim

        # --- Branch 1: 1D ResNet on raw IQ (time-domain) ---
        self.iq_branch = nn.Sequential(
            # Initial downsampling
            ConvBNReLU1D(2, 32, kernel_size=7, stride=2, padding=3),
            
            # Residual Blocks with strided convolutions (Learnable downsampling instead of MaxPool)
            ResidualBlock1D(32, 64, kernel_size=5, stride=2, padding=2),
            ResidualBlock1D(64, dim, kernel_size=3, stride=2, padding=1),
            
            SEBlock1D(dim), # Keep 1D attention for channel weighting
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # --- Branch 2: 2D ResNet on Spectrogram (frequency-domain) ---
        self.spec_branch = nn.Sequential(
            # Initial downsampling
            ConvBNReLU2D(1, 32, kernel_size=3, stride=2, padding=1),
            
            # Residual Blocks 
            ResidualBlock2D(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock2D(64, dim, kernel_size=3, stride=2, padding=1),
            
            # SOTA Attention for 2D (Spatial & Channel)
            CBAM(dim),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # --- Advanced Fusion Module ---
        self.gated_fusion = GatedFusion(dim)
        
        self.fusion_head = nn.Sequential(
            nn.Linear(dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(64, num_classes)

    def forward(self, iq: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iq: Raw IQ tensor [B, W, 2]
            spec: Spectrogram tensor [B, 1, F, T]

        Returns:
            Logits [B, num_classes]
        """
        # Branch 1: IQ
        iq_in = iq.permute(0, 2, 1)  # [B, 2, W]
        iq_feat = self.iq_branch(iq_in)  # [B, dim]

        # Branch 2: Spectrogram
        spec_feat = self.spec_branch(spec)  # [B, dim]

        # Gated Cross-Attention Fusion
        fused = self.gated_fusion(iq_feat, spec_feat)  # [B, dim]
        
        # Classification MLP
        features = self.fusion_head(fused)  # [B, 64]
        logits = self.output(features)  # [B, num_classes]

        return logits

    def get_features(self, iq: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """Extract 64-dim fused feature embedding (for t-SNE)."""
        iq_in = iq.permute(0, 2, 1)
        iq_feat = self.iq_branch(iq_in)
        spec_feat = self.spec_branch(spec)
        fused = self.gated_fusion(iq_feat, spec_feat)
        features = self.fusion_head(fused)
        return features

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SpectrogramBackboneClassifier(nn.Module):
    """
    Torchvision image backbones adapted for single-channel spectrogram input.

    The input spectrogram is upsampled to an ImageNet-like resolution and lifted
    from 1 to 3 channels so standard classification backbones can be reused
    without architecture-specific stem surgery.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = NUM_CLASSES,
        resize_to: Tuple[int, int] = (224, 224),
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.resize_to = resize_to
        self.input_adapter = nn.Sequential(
            nn.Upsample(size=resize_to, mode="bilinear", align_corners=False),
            nn.Conv2d(1, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.backbone, feature_dim = self._build_backbone(backbone_name)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _build_backbone(self, backbone_name: str) -> Tuple[nn.Module, int]:
        if backbone_name.startswith("resnet") or backbone_name.startswith("wide_resnet"):
            backbone = getattr(tv_models, backbone_name)(weights=None)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            return backbone, feature_dim

        if backbone_name.startswith("densenet"):
            backbone = getattr(tv_models, backbone_name)(weights=None)
            feature_dim = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
            return backbone, feature_dim

        if backbone_name.startswith("mobilenet"):
            backbone = getattr(tv_models, backbone_name)(weights=None)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            return backbone, feature_dim

        if backbone_name.startswith("efficientnet"):
            backbone = getattr(tv_models, backbone_name)(weights=None)
            feature_dim = backbone.classifier[-1].in_features
            backbone.classifier = nn.Identity()
            return backbone, feature_dim

        raise ValueError(f"Unsupported spectrogram backbone: {backbone_name}")

    def _extract_features(self, spec: torch.Tensor) -> torch.Tensor:
        spec = self.input_adapter(spec)
        features = self.backbone(spec)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._extract_features(spec))

    def get_features(self, spec: torch.Tensor) -> torch.Tensor:
        return self._extract_features(spec)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Model 5: DualBranchLite — Lightweight for Edge Devices
# ============================================================
class DualBranchLite(nn.Module):
    """
    Lightweight version of DualBranchFusionCNN for edge deployment.

    Key differences:
      - Fewer filters (16→32→64 instead of 32→64→128)
      - No SE blocks
      - Smaller fusion head
      - ~4x fewer parameters

    Designed for: Raspberry Pi, Jetson Nano, USRP companion boards
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()

        # Lightweight 1D branch
        self.iq_branch = nn.Sequential(
            ConvBNReLU1D(2, 16, kernel_size=7, stride=2, padding=3),
            ConvBNReLU1D(16, 32, kernel_size=5, stride=2, padding=2),
            ConvBNReLU1D(32, 64, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # Lightweight 2D branch
        self.spec_branch = nn.Sequential(
            ConvBNReLU2D(1, 16, kernel_size=3, stride=2, padding=1),
            ConvBNReLU2D(16, 32, kernel_size=3, stride=2, padding=1),
            ConvBNReLU2D(32, 64, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # Compact fusion
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, iq: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        iq_in = iq.permute(0, 2, 1)
        iq_feat = self.iq_branch(iq_in)
        spec_feat = self.spec_branch(spec)
        fused = torch.cat([iq_feat, spec_feat], dim=1)
        return self.classifier(fused)

    def get_features(self, iq: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        iq_in = iq.permute(0, 2, 1)
        iq_feat = self.iq_branch(iq_in)
        spec_feat = self.spec_branch(spec)
        return torch.cat([iq_feat, spec_feat], dim=1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Model 6: Custom DenseNet (User Contribution)
# ============================================================
class ComplexEncoder(nn.Module):
    """
    Encodes raw IQ [B, W, 2] -> [B, 4, W]: (I, Q, amplitude, phase).
    Preserves the complex-valued phase relationship lost by naive channel stacking.
    """
    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        iq = iq.permute(0, 2, 1)                        # [B, 2, W]
        I, Q = iq[:, 0:1], iq[:, 1:2]
        amp   = torch.sqrt(I ** 2 + Q ** 2)
        phase = torch.atan2(Q, I)
        return torch.cat([I, Q, amp, phase], dim=1)     # [B, 4, W]


class CBAM1D(nn.Module):
    """
    1D CBAM: Channel Attention + Spatial Attention.
    Symmetric counterpart to 2D CBAM used in the spectrogram branch.
    """
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.spatial_conv = nn.Conv1d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        avg = self.channel_fc(self.avg_pool(x).squeeze(-1))
        mx  = self.channel_fc(self.max_pool(x).squeeze(-1))
        ca  = self.sigmoid(avg + mx).unsqueeze(-1)
        x   = x * ca
        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        sa    = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


class CrossModalAttention(nn.Module):
    """
    Lightweight bidirectional cross-attention between IQ and spectrogram feature maps,
    applied BEFORE global pooling to preserve spatial structure during fusion.
    
    IQ   [B, C, L]    attends to Spectrogram [B, C, F*T]
    Spec [B, C, F*T]  attends to IQ          [B, C, L]
    """
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.iq_to_spec  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.spec_to_iq  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.iq_norm     = nn.LayerNorm(dim)
        self.spec_norm   = nn.LayerNorm(dim)

    def forward(
        self,
        iq_maps: torch.Tensor,      # [B, C, L]
        spec_maps: torch.Tensor,    # [B, C, F, T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, L       = iq_maps.shape
        _, _, F, T    = spec_maps.shape

        iq_seq   = iq_maps.permute(0, 2, 1)                        # [B, L,   C]
        spec_seq = spec_maps.flatten(2).permute(0, 2, 1)           # [B, F*T, C]

        # IQ attends to spectrogram (residual)
        iq_ctx,   _ = self.iq_to_spec(iq_seq,   spec_seq, spec_seq)
        iq_seq      = self.iq_norm(iq_seq + iq_ctx)

        # Spectrogram attends to IQ (residual)
        spec_ctx, _ = self.spec_to_iq(spec_seq, iq_seq,   iq_seq)
        spec_seq    = self.spec_norm(spec_seq + spec_ctx)

        iq_out   = iq_seq.permute(0, 2, 1)                         # [B, C, L]
        spec_out = spec_seq.permute(0, 2, 1).reshape(B, C, F, T)   # [B, C, F, T]
        return iq_out, spec_out


class DualBranchFusionCNN(nn.Module):
    """
    Dual-branch CNN with efficient concat-residual fusion for RF intrusion detection.

    Branch 1: compact 1D encoder on complex-encoded IQ [B, W, 2] + CBAM1D
    Branch 2: stronger depthwise-separable spectrogram encoder + SE/CBAM
    Fusion:   concat([iq_feat, spec_feat]) -> residual MLP projection -> logits
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        num_classes: int = NUM_CLASSES,
        branch_dim: int = 64,
        spec_branch_dim: Optional[int] = None,
        fusion_dim: int = 64,
        dropout: float = 0.15,
    ):
        super().__init__()

        assert branch_dim > 0, "branch_dim must be a positive integer"
        iq_dim = branch_dim
        spec_dim = spec_branch_dim or max(branch_dim, 80)

        # --- Branch 1: compact IQ encoder ---
        self.iq_encoder = ComplexEncoder()
        self.iq_branch = nn.Sequential(
            ConvBNReLU1D(4, 24, kernel_size=7, stride=2, padding=3),
            ResidualBlock1D(24, 48, kernel_size=5, stride=2, padding=2),
            ResidualBlock1D(48, iq_dim, kernel_size=3, stride=2, padding=1),
        )
        self.iq_attn = CBAM1D(iq_dim, reduction=8)
        self.iq_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())

        # --- Branch 2: stronger spectrogram encoder at lower cost ---
        self.spec_branch = nn.Sequential(
            ConvBNReLU2D(1, 24, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv2D(24, 48, stride=2),
            DepthwiseSeparableConv2D(48, 64, stride=1),
            DepthwiseSeparableConv2D(64, spec_dim, stride=2),
            SEBlock2D(spec_dim, reduction=8),
            ConvBNReLU2D(spec_dim, spec_dim, kernel_size=3, padding=1),
        )
        self.spec_attn = CBAM(spec_dim, ratio=8)
        self.spec_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        # --- Concat-residual fusion head ---
        self.fusion = ConcatResidualFusion(iq_dim, spec_dim, fusion_dim, dropout=dropout)
        self.fusion_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(64, num_classes)

    def _extract_features(self, iq: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        iq_maps = self.iq_attn(self.iq_branch(self.iq_encoder(iq)))
        spec_maps = self.spec_attn(self.spec_branch(spec))
        fused = self.fusion(self.iq_pool(iq_maps), self.spec_pool(spec_maps))
        return self.fusion_head(fused)

    def forward(self, iq: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            iq:   Raw IQ tensor       [B, W, 2]
            spec: Spectrogram tensor  [B, 1, F, T]
        Returns:
            Logits [B, num_classes]
        """
        return self.output(self._extract_features(iq, spec))

    def get_features(self, iq: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """Extract the fused embedding before the final classifier."""
        return self._extract_features(iq, spec)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# Model Registry
# ============================================================
EXTENDED_SPECTROGRAM_MODELS = [
    "resnet18_spec",
    "resnet34_spec",
    "resnet50_spec",
    "wide_resnet50_2_spec",
    "mobilenet_v2_spec",
    "mobilenet_v3_small_spec",
    "mobilenet_v3_large_spec",
    "efficientnet_b0_spec",
    "efficientnet_v2_s_spec",
    "densenet121_spec",
]

MODEL_REGISTRY = {
    "mlp_baseline": MLPBaseline,
    "cnn1d_iq": CNN1D_IQ,
    "cnn2d_spec": CNN2D_Spec,
    "dual_branch_fusion": DualBranchFusionCNN,
    "dual_branch_lite": DualBranchLite,
    "resnet18_spec": partial(SpectrogramBackboneClassifier, backbone_name="resnet18"),
    "resnet34_spec": partial(SpectrogramBackboneClassifier, backbone_name="resnet34"),
    "resnet50_spec": partial(SpectrogramBackboneClassifier, backbone_name="resnet50"),
    "wide_resnet50_2_spec": partial(
        SpectrogramBackboneClassifier,
        backbone_name="wide_resnet50_2",
    ),
    "mobilenet_v2_spec": partial(
        SpectrogramBackboneClassifier,
        backbone_name="mobilenet_v2",
    ),
    "mobilenet_v3_small_spec": partial(
        SpectrogramBackboneClassifier,
        backbone_name="mobilenet_v3_small",
    ),
    "mobilenet_v3_large_spec": partial(
        SpectrogramBackboneClassifier,
        backbone_name="mobilenet_v3_large",
    ),
    "efficientnet_b0_spec": partial(
        SpectrogramBackboneClassifier,
        backbone_name="efficientnet_b0",
    ),
    "efficientnet_v2_s_spec": partial(
        SpectrogramBackboneClassifier,
        backbone_name="efficientnet_v2_s",
    ),
    "densenet121_spec": partial(
        SpectrogramBackboneClassifier,
        backbone_name="densenet121",
    ),
}

DUAL_INPUT_MODELS = {"dual_branch_fusion", "dual_branch_lite"}
SPECTROGRAM_ONLY_MODELS = {"cnn2d_spec", *EXTENDED_SPECTROGRAM_MODELS}


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def is_dual_input(name: str) -> bool:
    """Check if a model requires both IQ and spectrogram inputs."""
    return name in DUAL_INPUT_MODELS


def is_spectrogram_only(name: str) -> bool:
    """Check if a model consumes only spectrogram input."""
    return name in SPECTROGRAM_ONLY_MODELS


def get_model_input_mode(name: str) -> str:
    """Return one of: 'iq', 'spectrogram', 'dual'."""
    if is_dual_input(name):
        return "dual"
    if is_spectrogram_only(name):
        return "spectrogram"
    return "iq"


# ============================================================
# Model Summary
# ============================================================
def print_model_summary():
    """Print parameter count comparison for all models."""
    print(f"\n{'='*60}")
    print(f"  Model Architecture Comparison")
    print(f"{'='*60}")
    print(f"  {'Model':<25} {'Params':>12} {'Input':>20}")
    print(f"  {'-'*25} {'-'*12} {'-'*20}")

    for name, cls in MODEL_REGISTRY.items():
        model = cls()
        n_params = model.count_params()
        input_mode = get_model_input_mode(name)
        input_type = {
            "dual": "IQ + Spec",
            "spectrogram": "Spec only",
            "iq": "IQ only",
        }[input_mode]
        print(f"  {name:<25} {n_params:>12,} {input_type:>20}")

    print(f"{'='*60}\n")


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    print_model_summary()

    batch_size = 4
    iq = torch.randn(batch_size, WINDOW_SIZE, 2)
    spec = torch.randn(batch_size, 1, SPEC_FREQ_BINS, SPEC_TIME_BINS)

    # Test each model
    for name, cls in MODEL_REGISTRY.items():
        model = cls()
        model.eval()
        input_mode = get_model_input_mode(name)

        with torch.no_grad():
            if input_mode == "dual":
                logits = model(iq, spec)
                features = model.get_features(iq, spec)
            elif input_mode == "spectrogram":
                logits = model(spec)
                features = model.get_features(spec)
            else:
                logits = model(iq)
                features = model.get_features(iq)

        print(f"  {name}: logits={logits.shape}, "
              f"features={features.shape}, "
              f"params={model.count_params():,}")

    print("\n✅ All models pass forward test.")
