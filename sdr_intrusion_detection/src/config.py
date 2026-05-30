"""
config.py — Centralized Configuration for SDR Intrusion Detection Pipeline
============================================================================
All hyperparameters, paths, model settings, and training defaults are
defined here in one place. Every other module imports from this file
instead of scattering magic numbers throughout the codebase.
"""
import os

# ============================================================
# PROJECT PATHS
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'unified_dataset'))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# ============================================================
# DATASET
# ============================================================
CLASS_NAMES = ['Machine', 'Human', 'Wildlife', 'Broadband_Jam', 'Narrowband_Jam', 'Benign']
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3                # Grayscale expanded to 3-ch for pretrained backbones
NORMALIZE_MEAN = [0.5, 0.5, 0.5]
NORMALIZE_STD = [0.5, 0.5, 0.5]

# ============================================================
# TRAINING DEFAULTS
# ============================================================
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-3
DEFAULT_OPTIMIZER = 'adamw'       # 'adamw' | 'sgd'
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_FOCAL_GAMMA = 2.0
DEFAULT_DROPOUT = 0.3
DEFAULT_PATIENCE = 7              # Early stopping patience (epochs without improvement)
DEFAULT_NUM_WORKERS = 4           # DataLoader workers

# ============================================================
# MODEL
# ============================================================
CUSTOM_MODEL_NAME = 'SDR_Custom_CoordASPP_Focal'

# Mapping of user-friendly model names to timm's registry names
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

# Custom model architecture constants
CUSTOM_PROJ_DIM = 512
CUSTOM_BACKBONE = 'resnet50'
CUSTOM_OUT_INDICES = (2, 3, 4)
CUSTOM_STAGE_DIMS = [512, 1024, 2048]
ASPP_DILATIONS = (1, 3, 6, 12)

# ============================================================
# GHOST-CAS ARCHITECTURE  (Experiment Variants)
# ============================================================
GHOST_RATIO = 2                    # Ghost cheap-op expansion ratio
GHOST_DW_SIZE = 3                  # Depthwise kernel for cheap operations
GHOST_CAS_PROJ_DIM = 256           # Halved projection channels for Ghost variants

# Ghost-CAS model names (used by get_model factory)
GHOSTCAS_RESNET_MODEL = 'SDR_GhostCAS_ResNet'    # Variant 1: ResNet50 + Ghost-CAS neck
GHOSTCAS_FULL_MODEL = 'SDR_GhostCAS_Full'        # Variant 2: Full custom (no ResNet50)

# Full Ghost-CAS stage definitions: (in_ch, mid_ch, out_ch, num_blocks, stride)
GHOST_STEM_CHANNELS = [32, 64]
GHOST_STAGE_CONFIGS = [
    (64,  96,  128, 2, 2),         # Stage 1: 56→28
    (128, 192, 256, 3, 2),         # Stage 2: 28→14
    (256, 384, 512, 3, 2),         # Stage 3: 14→7
]

# Models to compare in the Ghost-CAS experiment
GHOST_EXPERIMENT_MODELS = [
    CUSTOM_MODEL_NAME,             # Baseline: original CoordASPP + ResNet50
    GHOSTCAS_RESNET_MODEL,         # Variant 1: ResNet50 backbone + Ghost-CAS neck
    GHOSTCAS_FULL_MODEL,           # Variant 2: Full Ghost-CAS (no backbone)
]

# ============================================================
# ABLATION STUDY
# ============================================================
ABLATION_BACKBONE_EPOCHS = 50
ABLATION_BACKBONE_BATCH_SIZE = 64
ABLATION_HPARAM_EPOCHS = 50       # Was 30 — fixed to 50
ABLATION_CROSS_EPOCHS = 20
ABLATION_MAX_WORKERS = 2          # Parallel subprocess workers for Phase 1
ABLATION_N_TRIALS = 20            # Optuna trials for Phase 2
ABLATION_SUBPROCESS_TIMEOUT = 7200  # 2 hours max per model

# Optuna search space
OPTUNA_LR_CHOICES = [1e-4, 5e-4, 1e-3, 5e-3]
OPTUNA_BATCH_CHOICES = [16, 32, 64]
OPTUNA_OPTIMIZER_CHOICES = ['adamw', 'sgd']
OPTUNA_GAMMA_CHOICES = [1.0, 2.0, 3.0]
OPTUNA_WD_CHOICES = [1e-5, 1e-4, 1e-3]
OPTUNA_DROPOUT_CHOICES = [0.2, 0.3, 0.5]

# Cross-dataset directory names (relative to repo parent)
CROSS_DATASET_UNIFIED = 'unified_dataset'
CROSS_DATASET_RADAR = '_radar_staging'

# ============================================================
# PROFILING / BENCHMARKING
# ============================================================
PROFILE_WARMUP_RUNS = 5
PROFILE_BENCHMARK_RUNS = 50
BENCHMARK_N_RUNS = 100
BENCHMARK_WARMUP_RUNS = 10
ONNX_OPSET_VERSION = 14

# ============================================================
# WANDB
# ============================================================
WANDB_PROJECT = 'SDR-Intrusion-Detection'

# ============================================================
# LIVE INFERENCE (SDR Flowgraph)
# ============================================================
ZMQ_URL = 'tcp://127.0.0.1:5555'
SAMPLE_RATE = 1.92e6
WINDOW_SIZE = 1280
LIVE_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_resnet50.pth')

# ============================================================
# FIGURE GENERATION
# ============================================================
PLOT_COLORS = [
    '#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#264653',
    '#f4a261', '#606c38', '#bc6c25', '#8338ec', '#fb5607',
]
RADAR_CHART_TARGETS = [
    CUSTOM_MODEL_NAME, 'DenseNet121', 'ResNet50V2', 'MobileNetV2', 'EfficientNetV2S',
]
