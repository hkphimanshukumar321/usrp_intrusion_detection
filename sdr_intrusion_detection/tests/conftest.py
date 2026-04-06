import pytest
import torch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import WINDOW_SIZE, STFT_N_FFT, STFT_HOP

# Helper to calculate spectrogram dimensions to match compute_stft
F_BINS = STFT_N_FFT // 2 + 1
T_BINS = 1 + WINDOW_SIZE // STFT_HOP

@pytest.fixture
def dummy_iq_batch():
    """Returns a dummy batch of IQ data: [B, W, 2]"""
    batch_size = 4
    return torch.randn(batch_size, WINDOW_SIZE, 2)

@pytest.fixture
def dummy_spec_batch():
    """Returns a dummy batch of Spectrograms: [B, 1, F, T]"""
    batch_size = 4
    return torch.abs(torch.randn(batch_size, 1, F_BINS, T_BINS))

@pytest.fixture
def dummy_numpy_iq():
    """Returns a large dummy numpy IQ array mimicking a .dat file load."""
    return np.random.randn(5000) + 1j * np.random.randn(5000)
