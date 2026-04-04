#!/usr/bin/env python3
"""
data_loader.py — Load .dat IQ files and create PyTorch datasets.

Converts raw complex64 .dat files into windowed, labeled tensors suitable
for training CNN models.

Supports:
  - Simulated data (from sim_system.py)
  - BloodHound dataset (from download_bloodhound.py)
  - Custom .dat files from GNU Radio
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# ============================================================
# Constants
# ============================================================
WINDOW_SIZE = 256           # Samples per window
OVERLAP = 0.5               # 50% overlap
CLASS_NAMES = {0: "Clear", 1: "Human", 2: "Animal", 3: "Drone"}
NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# Raw .dat File Loading
# ============================================================
def load_dat_file(filepath: str) -> np.ndarray:
    """
    Load a GNU Radio complex64 .dat file.

    Args:
        filepath: Path to .dat file (complex64 format: interleaved float32 I/Q)

    Returns:
        Complex numpy array of shape [N]
    """
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
    Slice continuous IQ data into overlapping windows.

    Args:
        iq_data: Complex numpy array [N]
        window_size: Number of samples per window
        overlap: Fractional overlap between windows (0.0 to 0.99)

    Returns:
        Real-valued array of shape [num_windows, window_size, 2]
        where channel 0 = I (real), channel 1 = Q (imag)
    """
    step = int(window_size * (1 - overlap))
    num_windows = (len(iq_data) - window_size) // step + 1

    windows = np.zeros((num_windows, window_size, 2), dtype=np.float32)
    for i in range(num_windows):
        start = i * step
        segment = iq_data[start:start + window_size]
        windows[i, :, 0] = segment.real   # In-phase
        windows[i, :, 1] = segment.imag   # Quadrature

    return windows


def normalize_windows(windows: np.ndarray, method: str = "per_window") -> np.ndarray:
    """
    Normalize IQ windows.

    Args:
        windows: Array [N, W, 2]
        method: 'per_window' (each window to [-1,1]) or 'global' (across dataset)

    Returns:
        Normalized windows
    """
    if method == "per_window":
        # Normalize each window independently
        for i in range(len(windows)):
            max_val = np.max(np.abs(windows[i]))
            if max_val > 0:
                windows[i] /= max_val
    elif method == "global":
        max_val = np.max(np.abs(windows))
        if max_val > 0:
            windows /= max_val
    return windows


# ============================================================
# PyTorch Dataset
# ============================================================
class IQDataset(Dataset):
    """
    PyTorch Dataset for IQ window classification.

    Loads data from directory structure:
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
    ):
        """
        Args:
            data_dir: Directory containing .dat files
            window_size: Samples per window
            overlap: Fractional overlap between windows
            normalize: Normalization method ('per_window', 'global', or None)
            max_windows_per_class: Cap on windows per class (for quick testing)
            transform: Optional transform to apply to each sample
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

        # File→label mapping
        file_label_map = {
            "clear.dat": 0,
            "human.dat": 1,
            "animal.dat": 2,
            "drone.dat": 3,
        }

        all_windows = []
        all_labels = []

        print("\nLoading physical datasets into structured windows...")
        for filename, label in tqdm(file_label_map.items(), desc="Processing classes"):
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                print(f"  ⚠ {filename} not found, skipping class {label}")
                continue

            iq_data = load_dat_file(filepath)
            windows = iq_to_windows(iq_data, window_size, overlap)

            if max_windows_per_class and len(windows) > max_windows_per_class:
                # Random subsample
                idx = np.random.choice(len(windows), max_windows_per_class, replace=False)
                windows = windows[idx]

            labels = np.full(len(windows), label, dtype=np.int64)

            all_windows.append(windows)
            all_labels.append(labels)

            print(f"  Class {label} ({CLASS_NAMES[label]}): {len(windows):,} windows")

        if not all_windows:
            raise RuntimeError(f"No .dat files found in {data_dir}")

        self.windows = np.concatenate(all_windows, axis=0)  # [N, W, 2]
        self.labels = np.concatenate(all_labels, axis=0)     # [N]

        # Normalize
        if normalize:
            self.windows = normalize_windows(self.windows, method=normalize)

        # Convert to tensors
        self.windows_tensor = torch.from_numpy(self.windows)   # [N, W, 2]
        self.labels_tensor = torch.from_numpy(self.labels)     # [N]

        print(f"\n  Total: {len(self):,} windows, "
              f"shape={self.windows.shape}, classes={len(file_label_map)}")
        print(f"{'='*50}\n")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        window = self.windows_tensor[idx]  # [W, 2]
        label = self.labels_tensor[idx].item()

        if self.transform:
            window = self.transform(window)

        return window, label


# ============================================================
# Data Augmentation Transforms
# ============================================================
class IQAugmentation:
    """Data augmentation for IQ windows."""

    def __init__(
        self,
        add_noise_std: float = 0.01,
        circular_shift: bool = True,
        phase_rotation: bool = True,
    ):
        self.add_noise_std = add_noise_std
        self.circular_shift = circular_shift
        self.phase_rotation = phase_rotation

    def __call__(self, window: torch.Tensor) -> torch.Tensor:
        """
        Args:
            window: Tensor [W, 2] (I and Q channels)

        Returns:
            Augmented tensor [W, 2]
        """
        # Additive noise
        if self.add_noise_std > 0 and torch.rand(1) > 0.5:
            noise = torch.randn_like(window) * self.add_noise_std
            window = window + noise

        # Random circular shift
        if self.circular_shift and torch.rand(1) > 0.5:
            shift = torch.randint(0, window.shape[0], (1,)).item()
            window = torch.roll(window, shifts=shift, dims=0)

        # Random phase rotation (rotate I/Q plane)
        if self.phase_rotation and torch.rand(1) > 0.5:
            theta = torch.rand(1) * 2 * np.pi
            cos_t, sin_t = torch.cos(theta), torch.sin(theta)
            I, Q = window[:, 0], window[:, 1]
            window = torch.stack([
                I * cos_t - Q * sin_t,
                I * sin_t + Q * cos_t,
            ], dim=1)

        return window


# ============================================================
# K-Fold Cross-Validation Splitter
# ============================================================
def get_kfold_loaders(
    dataset: IQDataset,
    n_splits: int = 5,
    batch_size: int = 128,
    num_workers: int = 0,
    seed: int = 42,
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create stratified k-fold cross-validation DataLoader pairs.

    Args:
        dataset: IQDataset instance
        n_splits: Number of folds
        batch_size: Batch size
        num_workers: DataLoader workers
        seed: Random seed

    Returns:
        List of (train_loader, val_loader) tuples, one per fold
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = dataset.labels

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
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

    # Test single item
    window, label = ds[0]
    print(f"Sample: window={window.shape}, label={label} ({CLASS_NAMES[label]})")

    # Test k-fold
    folds = get_kfold_loaders(ds, n_splits=3, batch_size=64)
    for train_loader, val_loader in folds:
        batch_x, batch_y = next(iter(train_loader))
        print(f"  Batch: x={batch_x.shape}, y={batch_y.shape}")
        break
