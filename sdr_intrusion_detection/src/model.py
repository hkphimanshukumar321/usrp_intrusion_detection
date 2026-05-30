import math
import torch
import torch.nn as nn
import timm
from src.config import (
    NUM_CLASSES, TIMM_MODEL_MAP, CUSTOM_PROJ_DIM, CUSTOM_BACKBONE,
    CUSTOM_OUT_INDICES, CUSTOM_STAGE_DIMS, ASPP_DILATIONS, DEFAULT_DROPOUT,
    GHOST_RATIO, GHOST_DW_SIZE, GHOST_CAS_PROJ_DIM,
    GHOST_STEM_CHANNELS, GHOST_STAGE_CONFIGS,
    GHOSTCAS_RESNET_MODEL, GHOSTCAS_FULL_MODEL,
)


# ============================================================
# COORDINATE ATTENTION
# Decomposes spatial pooling into separate horizontal (time-axis)
# and vertical (frequency-axis) strips — maps directly onto
# the physical meaning of spectrogram axes.
# ============================================================
class CoordinateAttention(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=32):
        super().__init__()
        mid = max(8, in_ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1  = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(mid)
        self.act    = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, out_ch, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y   = torch.cat([x_h, x_w], dim=2)
        y   = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return x * a_h * a_w


# ============================================================
# GHOST MODULE  (Huawei, CVPR 2020)
# Generates feature maps in two stages:
#   1. A small set of "intrinsic" maps via standard conv
#   2. Additional "ghost" maps via cheap depthwise ops
# Result: same output shape, ~50% fewer params & FLOPs.
# ============================================================
class GhostModule(nn.Module):
    """Drop-in replacement for standard conv that halves parameters."""
    def __init__(self, in_ch, out_ch, kernel_size=1, ratio=GHOST_RATIO,
                 dw_size=GHOST_DW_SIZE, stride=1, relu=True):
        super().__init__()
        self.out_ch = out_ch
        init_ch = math.ceil(out_ch / ratio)
        new_ch = init_ch * (ratio - 1)

        # Primary convolution → intrinsic feature maps
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            *(nn.ReLU(inplace=True),) if relu else (),
        )

        # Cheap depthwise operation → ghost feature maps
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, new_ch, dw_size, 1, dw_size // 2,
                      groups=init_ch, bias=False),
            nn.BatchNorm2d(new_ch),
            *(nn.ReLU(inplace=True),) if relu else (),
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_ch]


# ============================================================
# GHOST BOTTLENECK  (Ghost Module + Coordinate Attention + Skip)
# The core building block of the full Ghost-CAS architecture.
# ============================================================
class GhostBottleneck(nn.Module):
    """
    Expand → (optional stride-2 DW conv) → Coordinate Attention → Squeeze.
    Skip connection with dimension matching when needed.
    """
    def __init__(self, in_ch, mid_ch, out_ch, dw_size=GHOST_DW_SIZE,
                 stride=1, use_ca=True):
        super().__init__()
        self.stride = stride

        # Ghost expand phase
        self.ghost1 = GhostModule(in_ch, mid_ch, relu=True)

        # Depthwise conv for spatial downsampling (only if stride > 1)
        if stride > 1:
            self.dw = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, dw_size, stride, dw_size // 2,
                          groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch),
            )
        else:
            self.dw = nn.Identity()

        # Coordinate Attention — preserves time/freq axis semantics
        self.ca = CoordinateAttention(mid_ch, mid_ch) if use_ca else nn.Identity()

        # Ghost squeeze phase (no ReLU — added after residual)
        self.ghost2 = GhostModule(mid_ch, out_ch, relu=False)

        # Skip connection with dimension/stride matching
        if in_ch != out_ch or stride > 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, dw_size, stride, dw_size // 2,
                          groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.ghost1(x)
        out = self.dw(out)
        out = self.ca(out)
        out = self.ghost2(out)
        return out + residual


# ============================================================
# GHOST ASPP  (Ghost-ified multi-scale module)
# Replaces LightASPP with Ghost Modules in each branch.
# ============================================================
class GhostASPP(nn.Module):
    """Multi-scale dilated convolutions using Ghost Modules for efficiency."""
    def __init__(self, in_ch, out_ch, dilations=ASPP_DILATIONS):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=d, dilation=d,
                          groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
                GhostModule(in_ch, out_ch, relu=True),
            ) for d in dilations
        ])
        self.fuse = GhostModule(out_ch * len(dilations), out_ch, relu=True)

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


# ============================================================
# LIGHT ASPP (original — kept for baseline model)
# ============================================================
class LightASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=ASPP_DILATIONS):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=d, dilation=d,
                          groups=in_ch, bias=False),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ) for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * len(dilations), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


# ============================================================
# BASELINE: SDR_Custom_CoordASPP_Focal  (original architecture)
# ============================================================
class SDR_Custom_CoordASPP_Focal(nn.Module):
    """
    Original architecture for the paper.
    Backbone : ResNet50 (multi-scale, stages 2-3-4)
    Attention : CoordinateAttention on each extracted stage
    Multi-scale: LightASPP on the deepest stage (stage 4)
    Fusion    : GAP per stage -> concat -> BN-stabilised classifier
    """
    _PROJ_DIM = CUSTOM_PROJ_DIM

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.features = timm.create_model(
            CUSTOM_BACKBONE, pretrained=pretrained,
            features_only=True, out_indices=CUSTOM_OUT_INDICES
        )
        self.feature_dim = self._PROJ_DIM * len(CUSTOM_OUT_INDICES)  # 1536

        stage_dims = CUSTOM_STAGE_DIMS
        p = self._PROJ_DIM

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, p, 1, bias=False),
                nn.BatchNorm2d(p),
                nn.ReLU(inplace=True)
            ) for d in stage_dims
        ])

        self.ca_s2 = CoordinateAttention(p, p)
        self.ca_s3 = CoordinateAttention(p, p)
        self.aspp      = LightASPP(p, p)
        self.attention = CoordinateAttention(p, p)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(DEFAULT_DROPOUT),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(DEFAULT_DROPOUT),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        s2, s3, s4 = self.features(x)
        s2 = self.proj[0](s2)
        s3 = self.proj[1](s3)
        s4 = self.proj[2](s4)
        s2 = self.ca_s2(s2)
        s3 = self.ca_s3(s3)
        s4 = self.attention(self.aspp(s4))
        v2 = torch.flatten(self.pool(s2), 1)
        v3 = torch.flatten(self.pool(s3), 1)
        v4 = torch.flatten(self.pool(s4), 1)
        x = torch.cat([v2, v3, v4], dim=1)
        return self.classifier(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# VARIANT 1: SDR_GhostCAS_ResNet
# ResNet50 backbone KEPT + Ghost-CAS neck (replaces proj + ASPP)
# ============================================================
class SDR_GhostCAS_ResNet(nn.Module):
    """
    Experiment Variant 1 — keeps the pretrained ResNet50 backbone
    but replaces the projection and ASPP layers with Ghost Modules
    and reduces PROJ_DIM from 512 → 256 for efficiency.
    """
    _PROJ_DIM = GHOST_CAS_PROJ_DIM  # 256

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.features = timm.create_model(
            CUSTOM_BACKBONE, pretrained=pretrained,
            features_only=True, out_indices=CUSTOM_OUT_INDICES
        )

        p = self._PROJ_DIM
        self.feature_dim = p * len(CUSTOM_OUT_INDICES)  # 768

        # Ghost projections (replace standard 1×1 conv)
        self.proj = nn.ModuleList([
            GhostModule(d, p, kernel_size=1, relu=True)
            for d in CUSTOM_STAGE_DIMS
        ])

        # Coordinate Attention on each stage
        self.ca_s2 = CoordinateAttention(p, p)
        self.ca_s3 = CoordinateAttention(p, p)

        # Ghost ASPP + CA on deepest stage
        self.aspp = GhostASPP(p, p)
        self.attention = CoordinateAttention(p, p)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(DEFAULT_DROPOUT),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        s2, s3, s4 = self.features(x)
        s2 = self.ca_s2(self.proj[0](s2))
        s3 = self.ca_s3(self.proj[1](s3))
        s4 = self.attention(self.aspp(self.proj[2](s4)))
        v2 = torch.flatten(self.pool(s2), 1)
        v3 = torch.flatten(self.pool(s3), 1)
        v4 = torch.flatten(self.pool(s4), 1)
        return self.classifier(torch.cat([v2, v3, v4], dim=1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# VARIANT 2: SDR_GhostCAS_Full
# NO ResNet50 backbone — fully custom Ghost-CAS architecture
# Purpose-built for RF spectrogram classification.
# ============================================================
class SDR_GhostCAS_Full(nn.Module):
    """
    Experiment Variant 2 — completely custom architecture.
    Stem → 3 Ghost-CAS stages → GhostASPP → Multi-scale fusion.

    Architecture:
        Stem:    Conv(3→32, s2) → Conv(32→64, s2)           → 56×56
        Stage 1: GhostBottleneck(64→128, s2)  × 2  + CA     → 28×28
        Stage 2: GhostBottleneck(128→256, s2) × 3  + CA     → 14×14
        Stage 3: GhostBottleneck(256→512, s2) × 3  + CA     →  7×7
        Neck:    GhostASPP(512→256) + CA
        Fusion:  GAP on stages 1,2,3 → concat (256×3=768) → classifier
    """
    _PROJ_DIM = GHOST_CAS_PROJ_DIM  # 256

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        p = self._PROJ_DIM
        stem_ch = GHOST_STEM_CHANNELS  # [32, 64]

        # Stem: 224→56  (two stride-2 convolutions)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_ch[0], stem_ch[1], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch[1]),
            nn.ReLU(inplace=True),
        )

        # Build Ghost-CAS stages from config
        stages = []
        for in_ch, mid_ch, out_ch, num_blocks, stride in GHOST_STAGE_CONFIGS:
            blocks = []
            # First block: stride-2 downsample + CA
            blocks.append(GhostBottleneck(in_ch, mid_ch, out_ch,
                                          stride=stride, use_ca=True))
            # Remaining blocks: stride-1, CA only on last block
            for i in range(1, num_blocks):
                use_ca = (i == num_blocks - 1)
                blocks.append(GhostBottleneck(out_ch, mid_ch, out_ch,
                                              stride=1, use_ca=use_ca))
            stages.append(nn.Sequential(*blocks))
        self.stage1, self.stage2, self.stage3 = stages

        # Stage output channels from config
        s1_ch = GHOST_STAGE_CONFIGS[0][2]  # 128
        s2_ch = GHOST_STAGE_CONFIGS[1][2]  # 256
        s3_ch = GHOST_STAGE_CONFIGS[2][2]  # 512

        # Multi-scale Ghost projections to uniform dim
        self.proj_s1 = GhostModule(s1_ch, p, relu=True)
        self.ca_s1   = CoordinateAttention(p, p)

        self.proj_s2 = GhostModule(s2_ch, p, relu=True)
        self.ca_s2   = CoordinateAttention(p, p)

        # Deepest stage: GhostASPP + CA
        self.aspp    = GhostASPP(s3_ch, p)
        self.ca_s3   = CoordinateAttention(p, p)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = p * 3  # 768

        self.classifier = nn.Sequential(
            nn.Dropout(DEFAULT_DROPOUT),
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        # Multi-scale fusion (same approach as baseline)
        f1 = self.ca_s1(self.proj_s1(s1))
        f2 = self.ca_s2(self.proj_s2(s2))
        f3 = self.ca_s3(self.aspp(s3))

        v1 = torch.flatten(self.pool(f1), 1)
        v2 = torch.flatten(self.pool(f2), 1)
        v3 = torch.flatten(self.pool(f3), 1)

        return self.classifier(torch.cat([v1, v2, v3], dim=1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# FACTORY
# ============================================================
def get_model(model_name: str, num_classes=NUM_CLASSES):
    """Returns the requested model architecture."""

    # --- Custom models ---
    if model_name == "SDR_Custom_CoordASPP_Focal":
        return SDR_Custom_CoordASPP_Focal(num_classes=num_classes)

    if model_name == GHOSTCAS_RESNET_MODEL:
        return SDR_GhostCAS_ResNet(num_classes=num_classes)

    if model_name == GHOSTCAS_FULL_MODEL:
        return SDR_GhostCAS_Full(num_classes=num_classes)

    # --- timm baselines ---
    if model_name in TIMM_MODEL_MAP:
        timm_name = TIMM_MODEL_MAP[model_name]
        try:
            model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
            model.count_params = lambda: sum(p.numel() for p in model.parameters() if p.requires_grad)
            return model
        except Exception as e:
            raise ValueError(f"Failed to load timm model '{timm_name}': {e}")

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Must be 'SDR_Custom_CoordASPP_Focal', '{GHOSTCAS_RESNET_MODEL}', "
        f"'{GHOSTCAS_FULL_MODEL}', or one of {list(TIMM_MODEL_MAP.keys())}"
    )
