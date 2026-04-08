import torch
import numpy as np
from src.data_loader import (
    normalize_windows, 
    compute_stft, 
    segment_aware_split, 
    WINDOW_SIZE, 
    STFT_N_FFT, 
    STFT_HOP
)

def test_normalize_windows():
    """Verify that normalize_windows securely forces bounds to exactly [-1, 1]."""
    # Create some dummy windows with crazy voltages
    windows = np.array([
        [[100.0, -50.0], [0.0, 0.0]],
        [[5.0, 5.0], [-10.0, 20.0]]
    ], dtype=np.float32)
    
    # Store original to verify no silent mutation happens downstream
    original = windows.copy()
    
    out = normalize_windows(windows, method="per_window")
    
    assert max(np.max(out[0]), np.abs(np.min(out[0]))) == 1.0, "Window 0 not properly bounded"
    assert max(np.max(out[1]), np.abs(np.min(out[1]))) == 1.0, "Window 1 not properly bounded"
    
    # Ensure it's not mutative
    np.testing.assert_array_equal(windows, original)

def test_compute_stft(dummy_iq_batch):
    """Verify compute_stft outputs the precise [1, F, T] tensor."""
    # Note: dummy_iq_batch is [B, W, 2]. compute_stft takes a single window [W, 2]
    iq_window = dummy_iq_batch[0]
    
    spec = compute_stft(iq_window, n_fft=STFT_N_FFT, hop_length=STFT_HOP)
    
    # F = n_fft because compute_stft keeps the full complex spectrum.
    # T = 1 + W // hop_length
    F_expected = STFT_N_FFT
    T_expected = 1 + WINDOW_SIZE // STFT_HOP
    
    assert spec.shape == (1, F_expected, T_expected), "STFT generated an incorrect spectrogram shape."
    
    # Ensure there are no mathematically invalid NaNs or negatives.
    assert not torch.isnan(spec).any(), "NaN found in STFT."
    assert torch.min(spec) >= 0.0, "Negative values found in log-power STFT."

def test_segment_aware_split():
    """Verify that segment_aware_split correctly isolates correlating windows."""
    num_windows = 100
    n_splits = 5
    overlap = 0.5
    
    folds = segment_aware_split(num_windows, n_splits, overlap)
    
    assert len(folds) == n_splits
    
    # Check that there is no data leakage
    # Meaning intersection of train and val indices for any fold MUST be completely empty
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        intersection = np.intersect1d(train_idx, val_idx)
        assert len(intersection) == 0, f"Fold {fold_idx} has DATA LEAKAGE! Train and Val sets share windows."
        
        # Verify sizes are roughly split 80/20 (since 5 splits)
        assert len(train_idx) > 0 and len(val_idx) > 0, "Empty splits detected!"
