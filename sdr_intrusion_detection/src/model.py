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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


NUM_CLASSES = 4
WINDOW_SIZE = 256
SPEC_FREQ_BINS = 128    # nfft
SPEC_TIME_BINS = 17     # depends on window_size, nperseg, noverlap


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
    Dual-branch CNN with feature fusion for RF intrusion detection.

    Branch 1: 1D-CNN on raw IQ [B, W, 2]
    Branch 2: 2D-CNN on STFT spectrogram [B, 1, F, T]
    Fusion:   Concatenate branch outputs → MLP → 4-class softmax

    This is our main contribution model: it combines time-domain (IQ)
    and frequency-domain (spectrogram) representations through late fusion,
    informed by the spike detection concept from the original GRC flowgraph.
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

        # --- Branch 1: 1D-CNN on raw IQ ---
        self.iq_branch = nn.Sequential(
            ConvBNReLU1D(2, 32, kernel_size=7, stride=1, padding=3),
            nn.MaxPool1d(2),
            ConvBNReLU1D(32, 64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool1d(2),
            ConvBNReLU1D(64, 128, kernel_size=3, stride=1, padding=1),
            SEBlock1D(128),
            nn.MaxPool1d(2),
            ConvBNReLU1D(128, branch1_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

        # --- Branch 2: 2D-CNN on Spectrogram ---
        self.spec_branch = nn.Sequential(
            ConvBNReLU2D(1, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBNReLU2D(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            ConvBNReLU2D(64, 128, kernel_size=3, padding=1),
            SEBlock2D(128),
            nn.MaxPool2d(2),
            ConvBNReLU2D(128, branch2_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        # --- Fusion Head ---
        fused_dim = branch1_dim + branch2_dim
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, fusion_dim),
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
        iq_feat = self.iq_branch(iq_in)  # [B, branch1_dim]

        # Branch 2: Spectrogram
        spec_feat = self.spec_branch(spec)  # [B, branch2_dim]

        # Late fusion
        fused = torch.cat([iq_feat, spec_feat], dim=1)  # [B, branch1+branch2]
        features = self.fusion_head(fused)  # [B, 64]
        logits = self.output(features)  # [B, num_classes]

        return logits

    def get_features(self, iq: torch.Tensor, spec: torch.Tensor) -> torch.Tensor:
        """Extract 64-dim fused feature embedding (for t-SNE)."""
        iq_in = iq.permute(0, 2, 1)
        iq_feat = self.iq_branch(iq_in)
        spec_feat = self.spec_branch(spec)
        fused = torch.cat([iq_feat, spec_feat], dim=1)
        features = self.fusion_head(fused)
        return features

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
class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(current_channels),
                nn.Conv2d(current_channels, growth_rate, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.layers.append(layer)
            current_channels += growth_rate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all previous features
            concat_features = torch.cat(features, dim=1)
            new_feature = layer(concat_features)
            features.append(new_feature)
        return torch.cat(features, dim=1)

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, compression):
        super().__init__()
        reduced_filters = max(1, int(in_channels * compression))
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, reduced_filters, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class CustomDenseNet(nn.Module):
    """
    Lightweight DenseNet architecture ported from TensorFlow configuration.
    Inputs: 2D Spectrograms [B, 1, F, T]
    """
    def __init__(self, in_channels=1, num_classes=NUM_CLASSES, growth_rate=8, compression=0.5):
        super().__init__()
        num_dense_layers = [3, 3, 3]
        
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, growth_rate * 2, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        
        current_channels = growth_rate * 2
        self.blocks = nn.ModuleList()
        
        # Dense + Transition Blocks
        for num_layers in num_dense_layers:
            db = DenseBlock(current_channels, num_layers, growth_rate)
            self.blocks.append(db)
            current_channels += num_layers * growth_rate
            
            tb = TransitionBlock(current_channels, compression)
            self.blocks.append(tb)
            current_channels = max(1, int(current_channels * compression))
            
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(current_channels, num_classes)
        )
        
    def forward(self, x):
        """x: [B, 1, F, T] Spectrogram"""
        x = self.init_conv(x)
        features = x
        for block in self.blocks:
            features = block(features)
        return self.classifier(features)
        
    def get_features(self, x):
        x = self.init_conv(x)
        features = x
        for block in self.blocks:
            features = block(features)
        features = F.adaptive_avg_pool2d(features, 1)
        return features.flatten(1)
        
    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ============================================================
# Model Registry
# ============================================================
MODEL_REGISTRY = {
    "mlp_baseline": MLPBaseline,
    "cnn1d_iq": CNN1D_IQ,
    "cnn2d_spec": CNN2D_Spec,
    "custom_densenet": CustomDenseNet,
    "dual_branch_fusion": DualBranchFusionCNN,
    "dual_branch_lite": DualBranchLite,
}

DUAL_INPUT_MODELS = {"dual_branch_fusion", "dual_branch_lite"}


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. "
                         f"Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def is_dual_input(name: str) -> bool:
    """Check if a model requires both IQ and spectrogram inputs."""
    return name in DUAL_INPUT_MODELS


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
        input_type = "IQ + Spec" if name in DUAL_INPUT_MODELS else "IQ only"
        if name in ["cnn2d_spec", "custom_densenet"]:
            input_type = "Spec only"
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

        with torch.no_grad():
            if name in DUAL_INPUT_MODELS:
                logits = model(iq, spec)
                features = model.get_features(iq, spec)
            elif name in ["cnn2d_spec", "custom_densenet"]:
                logits = model(spec)
                features = model.get_features(spec)
            else:
                logits = model(iq)
                features = model.get_features(iq)

        print(f"  {name}: logits={logits.shape}, "
              f"features={features.shape}, "
              f"params={model.count_params():,}")

    print("\n✅ All models pass forward test.")
