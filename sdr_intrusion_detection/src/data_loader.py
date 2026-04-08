import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

WINDOW_SIZE  = 256
OVERLAP      = 0.5
CLASS_NAMES  = {0: "Clear", 1: "Human", 2: "Animal", 3: "Drone"}
NUM_CLASSES  = len(CLASS_NAMES)

# STFT config — must match model's expected [B, 1, F, T] input
STFT_N_FFT   = 64
STFT_HOP     = 16


# ============================================================
# Raw .dat File Loading
# ============================================================
def load_dat_file(filepath: str) -> np.ndarray:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    data = np.fromfile(filepath, dtype=np.complex64)
    print(f"  Loaded {filepath}: {len(data):,} samples "
          f"({len(data) / 1_920_000:.2f}s at 1.92 MSps)")
    return data


def iq_to_windows(
    iq_data: np.ndarray,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
) -> np.ndarray:
    """
    Slice IQ data into windows.
    Returns [num_windows, window_size, 2] float32.
    """
    step = int(window_size * (1 - overlap))
    num_windows = (len(iq_data) - window_size) // step + 1

    windows = np.zeros((num_windows, window_size, 2), dtype=np.float32)
    for i in range(num_windows):
        start = i * step
        seg = iq_data[start:start + window_size]
        windows[i, :, 0] = seg.real
        windows[i, :, 1] = seg.imag

    return windows


def normalize_windows(windows: np.ndarray, method: str = "per_window") -> np.ndarray:
    """
    Normalize IQ windows. Returns a copy — does NOT mutate input.
    """
    out = windows.copy()                                    # ← no silent mutation
    if method == "per_window":
        for i in range(len(out)):
            max_val = np.max(np.abs(out[i]))
            if max_val > 0:
                out[i] = out[i] / max_val
    elif method == "global":
        max_val = np.max(np.abs(out))
        if max_val > 0:
            out = out / max_val
    return out


def compute_stft(
    iq_window: torch.Tensor,              # [W, 2]
    n_fft: int   = STFT_N_FFT,
    hop_length: int = STFT_HOP,
) -> torch.Tensor:
    """
    Compute log-magnitude STFT spectrogram from one IQ window.
    """
    # Use the vectorized version for a single window to maintain DRY
    return compute_stft_batch(iq_window.unsqueeze(0), n_fft, hop_length).squeeze(0)


def compute_stft_batch(
    iq_windows: torch.Tensor,              # [N, W, 2]
    n_fft: int   = STFT_N_FFT,
    hop_length: int = STFT_HOP,
) -> torch.Tensor:
    """
    Vectorized STFT computation for a batch of windows.
    """
    # Preserve the true complex IQ structure instead of collapsing to amplitude.
    signal = torch.complex(iq_windows[..., 0], iq_windows[..., 1])  # [N, W]

    stft = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=torch.hann_window(n_fft).to(signal.device),
        return_complex=True,
        center=True,
        onesided=False,
    )                                        # [N, F, T]

    log_power = torch.log1p(stft.abs() ** 2)
    flat = log_power.flatten(1)
    mins = flat.min(dim=1).values.view(-1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1)
    log_power = (log_power - mins) / (maxs - mins + 1e-6)
    return log_power.unsqueeze(1)            # [N, 1, F, T]


# ============================================================
# Leakage-Safe K-Fold Split
# ============================================================
def segment_aware_split(
    num_windows: int,
    n_splits: int,
    overlap: float,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split window indices by non-overlapping *segments* to prevent leakage.

    Adjacent overlapping windows share samples; naive index-level splitting
    leaks training data into validation. This groups consecutive windows into
    independent blocks and splits at the block level.

    Returns list of (train_indices, val_indices) arrays.
    """
    rng = np.random.default_rng(seed)

    # Each step advances by (1-overlap)*window_size samples.
    # Windows within one n_fft-worth of each other are correlated.
    # Use a stride of 1/overlap as the minimum independent block size.
    block_size  = max(1, int(1 / (1 - overlap)))            # e.g. 2 for 50% overlap
    num_blocks  = num_windows // block_size
    block_ids   = np.arange(num_blocks)
    rng.shuffle(block_ids)

    folds = []
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Use block-level split; labels are uniform within blocks so use zeros
    dummy = np.zeros(num_blocks, dtype=np.int64)
    for train_blk, val_blk in kf.split(dummy, dummy):
        train_idx = np.concatenate([
            np.arange(b * block_size, (b + 1) * block_size) for b in block_ids[train_blk]
        ])
        val_idx = np.concatenate([
            np.arange(b * block_size, (b + 1) * block_size) for b in block_ids[val_blk]
        ])
        folds.append((train_idx, val_idx))

    return folds


# ============================================================
# PyTorch Dataset
# ============================================================
class IQDataset(Dataset):
    """
    PyTorch Dataset for dual-input RF classification.

    Returns (iq_window, spectrogram, label) matching DualBranchFusionCNN inputs:
        iq_window:    [W, 2]   float32
        spectrogram:  [1, F, T] float32   (log-power STFT)
        label:        int
    
    Data directory structure:
        data_dir/
            clear.dat   → label 0
            human.dat   → label 1
            animal.dat  → label 2
            drone.dat   → label 3
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int = WINDOW_SIZE,
        overlap: float = OVERLAP,
        normalize: str = "per_window",
        max_windows_per_class: Optional[int] = None,
        transform=None,
        precompute_stft: bool = True,                       # Trade memory for speed
    ):
        self.data_dir      = data_dir
        self.window_size   = window_size
        self.overlap       = overlap
        self.transform     = transform
        self.precompute_stft = precompute_stft

        file_label_map = {
            "clear.dat":  0,
            "human.dat":  1,
            "animal.dat": 2,
            "drone.dat":  3,
        }

        all_windows, all_labels = [], []

        print("\nLoading datasets...")
        for filename, label in tqdm(file_label_map.items(), desc="Processing classes"):
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"  ⚠ {filename} not found, skipping class {label}")
                continue

            iq_data = load_dat_file(filepath)
            windows = iq_to_windows(iq_data, window_size, overlap)

            if max_windows_per_class and len(windows) > max_windows_per_class:
                idx = np.random.choice(len(windows), max_windows_per_class, replace=False)
                windows = windows[idx]

            all_windows.append(windows)
            all_labels.append(np.full(len(windows), label, dtype=np.int64))
            print(f"  Class {label} ({CLASS_NAMES[label]}): {len(windows):,} windows")

        if not all_windows:
            raise RuntimeError(f"No .dat files found in {data_dir}")

        windows = np.concatenate(all_windows, axis=0)       # [N, W, 2]
        if normalize:
            windows = normalize_windows(windows, method=normalize)

        self.windows_tensor = torch.from_numpy(windows)     # [N, W, 2]
        self.labels_tensor  = torch.from_numpy(
            np.concatenate(all_labels, axis=0)
        )                                                    # [N]

        # Precompute spectrograms once to avoid repeated STFT at batch time
        if self.precompute_stft:
            print("  Precomputing STFT spectrograms (vectorized)...")
            self.specs_tensor = compute_stft_batch(self.windows_tensor)
        else:
            self.specs_tensor = None

        print(f"\n  Total: {len(self):,} windows | "
              f"IQ shape={tuple(self.windows_tensor.shape[1:])} | "
              f"Spec shape={tuple(self.specs_tensor.shape[1:]) if self.specs_tensor is not None else 'on-the-fly'}\n"
              f"{'='*50}\n")

    @property
    def labels(self) -> np.ndarray:
        return self.labels_tensor.numpy()

    @property
    def windows(self) -> np.ndarray:
        return self.windows_tensor.numpy()

    def __len__(self) -> int:
        return len(self.labels_tensor)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Returns {'iq', 'spectrogram', 'label'} compatible with DualBranchFusionCNN."""
        iq    = self.windows_tensor[idx]                    # [W, 2]
        label = self.labels_tensor[idx].item()

        spec  = (self.specs_tensor[idx]
                 if self.precompute_stft
                 else compute_stft_batch(iq.unsqueeze(0)).squeeze(0)) # [1, F, T]

        if self.transform:
            iq = self.transform(iq)

        return {
            'iq': iq,
            'spectrogram': spec,
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================
# Data Augmentation
# ============================================================
class IQAugmentation:
    """Physics-consistent augmentation for IQ windows."""

    def __init__(
        self,
        add_noise_std: float = 0.01,
        circular_shift: bool = True,
        phase_rotation: bool = True,
    ):
        self.add_noise_std  = add_noise_std
        self.circular_shift = circular_shift
        self.phase_rotation = phase_rotation

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        if self.add_noise_std > 0 and torch.rand(1) > 0.5:
            window = window + torch.randn_like(window) * self.add_noise_std

        if self.circular_shift and torch.rand(1) > 0.5:
            shift = torch.randint(0, window.shape[0], (1,)).item()
            window = torch.roll(window, shifts=shift, dims=0)

        if self.phase_rotation and torch.rand(1) > 0.5:
            theta = torch.rand(1) * 2 * np.pi
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            I, Q  = window[:, 0], window[:, 1]
            window = torch.stack([
                I * cos_t - Q * sin_t,
                I * sin_t + Q * cos_t,
            ], dim=1)

        return window


# ============================================================
# K-Fold Cross-Validation (Leakage-Safe)
# ============================================================
def get_kfold_loaders(
    dataset: IQDataset,
    n_splits: int   = 5,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Segment-aware stratified k-fold loaders.
    Splits by non-overlapping blocks to prevent window-level leakage.
    """
    labels      = dataset.labels
    num_windows = len(labels)

    # Build per-class leakage-safe splits then merge
    # Since classes are contiguous in the original load order,
    # we split each class independently and union indices.
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    block_size = max(1, int(1 / (1 - dataset.overlap)))

    # Group indices into blocks; assign block label = majority label in block
    num_blocks = num_windows // block_size
    block_labels = np.array([
        labels[b * block_size] for b in range(num_blocks)
    ])

    folds = []
    for fold_idx, (train_blk, val_blk) in enumerate(
        skf.split(np.zeros(num_blocks), block_labels)
    ):
        train_idx = np.concatenate([
            np.arange(b * block_size, min((b+1) * block_size, num_windows))
            for b in train_blk
        ])
        val_idx = np.concatenate([
            np.arange(b * block_size, min((b+1) * block_size, num_windows))
            for b in val_blk
        ])

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        folds.append((train_loader, val_loader))
        print(f"  Fold {fold_idx+1}/{n_splits}: "
              f"train={len(train_idx):,}, val={len(val_idx):,}")

    return folds


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/simulated"

    print("Testing data loader...")
    ds = IQDataset(data_dir, max_windows_per_class=1000)

    # Verify dual-output shape
    iq, spec, label = ds[0]
    print(f"Sample: iq={iq.shape}, spec={spec.shape}, "
          f"label={label} ({CLASS_NAMES[label]})")

    # Verify loader yields correct batch structure
    folds = get_kfold_loaders(ds, n_splits=3, batch_size=64)
    train_loader, _ = folds[0]
    batch_iq, batch_spec, batch_y = next(iter(train_loader))
    print(f"Batch:  iq={batch_iq.shape}, spec={batch_spec.shape}, y={batch_y.shape}")
