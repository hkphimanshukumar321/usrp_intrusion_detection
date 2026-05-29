import torch
import torch.nn as nn
import timm
from src.data_loader import NUM_CLASSES

# Mapping of the user's requested model names to timm's official repository names
TIMM_MODEL_MAP = {
    'DenseNet121': 'densenet121',
    'DenseNet169': 'densenet169',
    'DenseNet201': 'densenet201',
    'MobileNetV2': 'mobilenetv2_100',
    'InceptionV3': 'inception_v3',
    'InceptionResNetV2': 'inception_resnet_v2',
    'NASNetMobile': 'mnasnet_100',
    'ResNet50V2': 'resnetv2_50',
    'ResNet101V2': 'resnetv2_101',
    'ResNet152V2': 'resnetv2_152',
    'Xception': 'legacy_xception',
    'VGG16': 'vgg16',
    'VGG19': 'vgg19',
    'EfficientNetV2L': 'tf_efficientnetv2_l',
    'EfficientNetV2M': 'tf_efficientnetv2_m',
    'EfficientNetV2S': 'tf_efficientnetv2_s',
    'MobileNetV3Large': 'mobilenetv3_large_100',
    'MobileNetV3Small': 'mobilenetv3_small_100',
}


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
# LIGHT ASPP (stripped depthwise variant)
# Captures multi-scale RF spectral patterns at the deepest
# stage without segmentation-grade overhead.
# ============================================================
class LightASPP(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 3, 6, 12)):
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
# CUSTOM ARCHITECTURE: SDR_Custom_CoordASPP_Focal
# ============================================================
class SDR_Custom_CoordASPP_Focal(nn.Module):
    """
    Our custom proprietary architecture for the paper.
    Backbone : ResNet50 (multi-scale, stages 2-3-4)
    Attention : CoordinateAttention on each extracted stage
    Multi-scale: LightASPP on the deepest stage (stage 4)
    Fusion    : GAP per stage -> concat -> BN-stabilised classifier
    """
    _PROJ_DIM = 512

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        self.features = timm.create_model(
            'resnet50', pretrained=pretrained,
            features_only=True, out_indices=(2, 3, 4)
        )
        self.feature_dim = self._PROJ_DIM * 3  # 1536 after concat

        stage_dims = [512, 1024, 2048]
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
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
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
# FACTORY
# ============================================================
def get_model(model_name: str, num_classes=NUM_CLASSES):
    """Returns either our custom model or one of the standard timm baselines."""
    if model_name == "SDR_Custom_CoordASPP_Focal":
        return SDR_Custom_CoordASPP_Focal(num_classes=num_classes)

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
        f"Must be 'SDR_Custom_CoordASPP_Focal' or one of {list(TIMM_MODEL_MAP.keys())}"
    )
