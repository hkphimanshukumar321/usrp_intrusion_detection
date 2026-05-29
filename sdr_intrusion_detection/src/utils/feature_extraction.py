#!/usr/bin/env python3
"""
feature_extraction.py — Convert raw IQ windows into multiple representations.

Produces three feature types from each IQ window:
  1. Raw IQ:         [window_size, 2]     — direct I/Q channels
  2. STFT Spectrogram: [n_freq, n_time]   — time-frequency representation
  3. Amplitude Envelope: [window_size, 1] — |I + jQ| magnitude

These are used by the dual-branch CNN and for visualization.
"""

import numpy as np
import torch
from scipy import signal as sp_signal
from typing import Tuple, Optional
from tqdm import tqdm


# ============================================================
# STFT Spectrogram
# ============================================================
def compute_spectrogram(
    iq_window: np.ndarray,
    nperseg: int = 64,
    noverlap: int = 48,
    nfft: int = 128,
    fs: float = 1.92e6,
    log_scale: bool = True,
) -> np.ndarray:
    """
    Compute STFT spectrogram from a complex IQ window.

    Args:
        iq_window: Complex array [window_size] or real array [window_size, 2]
        nperseg: STFT segment length
        noverlap: STFT overlap
        nfft: FFT size (determines frequency resolution)
        fs: Sample rate
        log_scale: If True, convert to log power (dB)

    Returns:
        Spectrogram array [n_freq, n_time] (real-valued)
    """
    # Convert [W, 2] to complex if needed
    if iq_window.ndim == 2 and iq_window.shape[1] == 2:
        iq_complex = iq_window[:, 0] + 1j * iq_window[:, 1]
    else:
        iq_complex = iq_window

    _, _, Zxx = sp_signal.stft(
        iq_complex,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
    )

    # Power spectrogram
    power = np.abs(Zxx) ** 2

    if log_scale:
        power = 10 * np.log10(power + 1e-12)  # Avoid log(0)
        # Normalize to [0, 1]
        power = (power - power.min()) / (power.max() - power.min() + 1e-12)

    return power.astype(np.float32)


def compute_spectrogram_batch(
    iq_windows: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Compute spectrograms for a batch of IQ windows.

    Args:
        iq_windows: Array [N, W, 2]

    Returns:
        Spectrograms [N, n_freq, n_time]
    """
    specs = []
    for i in tqdm(range(len(iq_windows)), desc="Computing spectrograms"):
        spec = compute_spectrogram(iq_windows[i], **kwargs)
        specs.append(spec)

    return np.stack(specs, axis=0)


# ============================================================
# Amplitude Envelope
# ============================================================
def compute_amplitude_envelope(iq_window: np.ndarray) -> np.ndarray:
    """
    Compute the amplitude envelope (magnitude) of an IQ window.

    Args:
        iq_window: Array [window_size, 2] (I and Q channels)

    Returns:
        Amplitude envelope [window_size, 1]
    """
    I = iq_window[:, 0]
    Q = iq_window[:, 1]
    magnitude = np.sqrt(I ** 2 + Q ** 2)
    return magnitude.reshape(-1, 1).astype(np.float32)


def compute_amplitude_batch(iq_windows: np.ndarray) -> np.ndarray:
    """
    Compute amplitude envelopes for a batch.

    Args:
        iq_windows: Array [N, W, 2]

    Returns:
        Envelopes [N, W, 1]
    """
    I = iq_windows[:, :, 0]
    Q = iq_windows[:, :, 1]
    magnitude = np.sqrt(I ** 2 + Q ** 2)
    return magnitude[:, :, np.newaxis].astype(np.float32)


# ============================================================
# Instantaneous Phase and Frequency
# ============================================================
def compute_instantaneous_phase(iq_window: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous phase from IQ data.

    Args:
        iq_window: Array [window_size, 2]

    Returns:
        Instantaneous phase [window_size, 1] in radians
    """
    I = iq_window[:, 0]
    Q = iq_window[:, 1]
    phase = np.arctan2(Q, I)
    return phase.reshape(-1, 1).astype(np.float32)


def compute_instantaneous_frequency(iq_window: np.ndarray, fs: float = 1.92e6) -> np.ndarray:
    """
    Compute instantaneous frequency (derivative of phase).

    Args:
        iq_window: Array [window_size, 2]
        fs: Sample rate

    Returns:
        Instantaneous frequency [window_size-1, 1] in Hz
    """
    phase = compute_instantaneous_phase(iq_window).flatten()
    # Unwrap phase to avoid discontinuities
    phase_unwrapped = np.unwrap(phase)
    # Frequency = d(phase)/dt / (2*pi)
    freq = np.diff(phase_unwrapped) * fs / (2 * np.pi)
    return freq.reshape(-1, 1).astype(np.float32)


# ============================================================
# Handcrafted Statistical Features (for SVM baseline)
# ============================================================
def compute_statistical_features(iq_window: np.ndarray) -> np.ndarray:
    """
    Compute handcrafted statistical features for traditional ML baselines.

    Features (20 total):
      - Mean, std, max, min of I and Q channels (8)
      - Mean, std, max, min of amplitude (4)
      - Mean, std of instantaneous phase (2)
      - Mean, std of instantaneous frequency (2)
      - Kurtosis and skewness of amplitude (2)
      - Peak-to-average power ratio (PAPR) (1)
      - Zero-crossing rate of I channel (1)

    Args:
        iq_window: Array [window_size, 2]

    Returns:
        Feature vector [20]
    """
    from scipy.stats import kurtosis, skew

    I = iq_window[:, 0]
    Q = iq_window[:, 1]
    amp = np.sqrt(I ** 2 + Q ** 2)
    phase = np.arctan2(Q, I)
    freq = np.diff(np.unwrap(phase))

    features = []

    # I/Q statistics (8)
    for ch in [I, Q]:
        features.extend([np.mean(ch), np.std(ch), np.max(ch), np.min(ch)])

    # Amplitude statistics (4)
    features.extend([np.mean(amp), np.std(amp), np.max(amp), np.min(amp)])

    # Phase statistics (2)
    features.extend([np.mean(phase), np.std(phase)])

    # Frequency statistics (2)
    features.extend([np.mean(freq), np.std(freq)])

    # Higher-order statistics (2)
    features.extend([kurtosis(amp), skew(amp)])

    # PAPR (1)
    papr = np.max(amp ** 2) / (np.mean(amp ** 2) + 1e-12)
    features.append(papr)

    # Zero-crossing rate (1)
    zcr = np.sum(np.abs(np.diff(np.sign(I)))) / (2 * len(I))
    features.append(zcr)

    return np.array(features, dtype=np.float32)


def compute_statistical_features_batch(iq_windows: np.ndarray) -> np.ndarray:
    """
    Compute statistical features for a full batch.

    Args:
        iq_windows: Array [N, W, 2]

    Returns:
        Features [N, 20]
    """
    features = []
    for i in tqdm(range(len(iq_windows)), desc="Computing statistical features"):
        feat = compute_statistical_features(iq_windows[i])
        features.append(feat)
    return np.stack(features, axis=0)


# ============================================================
# PyTorch Multi-Representation Dataset Wrapper
# ============================================================
class MultiRepresentationDataset(torch.utils.data.Dataset):
    """
    Wraps an IQDataset and computes spectrograms on-the-fly.

    Returns a dict with keys: 'iq', 'spectrogram', 'label'
    """

    def __init__(self, iq_dataset, precompute_specs: bool = True):
        """
        Args:
            iq_dataset: IQDataset instance
            precompute_specs: If True, compute all spectrograms upfront (faster training)
        """
        self.iq_dataset = iq_dataset
        self.precompute_specs = precompute_specs

        if precompute_specs:
            print("  Precomputing spectrograms...")
            self.spectrograms = compute_spectrogram_batch(iq_dataset.windows)
            print(f"  Spectrograms shape: {self.spectrograms.shape}")
            self.spectrograms = torch.from_numpy(self.spectrograms)

    def __len__(self):
        return len(self.iq_dataset)

    def __getitem__(self, idx):
        iq_window, label = self.iq_dataset[idx]

        if self.precompute_specs:
            spec = self.spectrograms[idx]
        else:
            window_np = iq_window.numpy()
            spec = torch.from_numpy(compute_spectrogram(window_np))

        return {
            'iq': iq_window,                    # [W, 2]
            'spectrogram': spec.unsqueeze(0),   # [1, F, T] (add channel dim)
            'label': label,
        }


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    window = np.random.randn(256, 2).astype(np.float32)

    print("Spectrogram:", compute_spectrogram(window).shape)
    print("Amplitude:", compute_amplitude_envelope(window).shape)
    print("Phase:", compute_instantaneous_phase(window).shape)
    print("Frequency:", compute_instantaneous_frequency(window).shape)
    print("Statistical features:", compute_statistical_features(window).shape)

    # Batch test
    batch = np.random.randn(10, 256, 2).astype(np.float32)
    print("Batch specs:", compute_spectrogram_batch(batch).shape)
    print("Batch amps:", compute_amplitude_batch(batch).shape)
    print("Batch stats:", compute_statistical_features_batch(batch).shape)
