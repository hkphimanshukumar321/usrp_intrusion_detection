#!/usr/bin/env python3
"""
spike_detector.py — Python port of the GRC spike detection chain.

Replicates the flowgraph:
    Complex→Float → PeakDetector → Threshold → BinarySlicer

This serves as the traditional (non-ML) baseline detector.
"""

import numpy as np
from typing import Tuple


# ============================================================
# Peak Detector (replicates gr::blocks::peak_detector_fb)
# ============================================================
def peak_detector(
    signal: np.ndarray,
    threshold_rise: float = 10.0,
    threshold_fall: float = 20.0,
    look_ahead: int = 10,
    alpha: float = 0.001,
) -> np.ndarray:
    """
    Detect peaks in a signal using a running average threshold.

    Mimics GNU Radio's peak_detector_fb block:
      - Maintains a running average of the signal
      - Flags a peak when signal exceeds (running_avg * threshold_rise)
      - Unflag when signal drops below (running_avg * threshold_fall)

    Args:
        signal: Input real-valued signal [N]
        threshold_rise: Peak detection rising threshold factor
        threshold_fall: Peak detection falling threshold factor
        look_ahead: Number of samples to look ahead for peak confirmation
        alpha: Running average smoothing factor

    Returns:
        Binary peak indicator [N] (1 = peak, 0 = no peak)
    """
    N = len(signal)
    peaks = np.zeros(N, dtype=np.float32)
    running_avg = np.abs(signal[0]) if N > 0 else 0.0
    in_peak = False

    for i in range(N):
        val = np.abs(signal[i])
        running_avg = alpha * val + (1 - alpha) * running_avg

        if not in_peak:
            if val > running_avg * threshold_rise:
                # Check look-ahead: is this sample the local max?
                is_local_max = True
                for j in range(1, min(look_ahead + 1, N - i)):
                    if np.abs(signal[i + j]) > val:
                        is_local_max = False
                        break
                if is_local_max:
                    peaks[i] = 1.0
                    in_peak = True
        else:
            if val < running_avg * threshold_fall:
                in_peak = False

    return peaks


# ============================================================
# Threshold Block (replicates gr::blocks::threshold_ff)
# ============================================================
def threshold_block(
    signal: np.ndarray,
    low: float = -20.0,
    high: float = 20.0,
) -> np.ndarray:
    """
    Binary threshold with hysteresis.

    Args:
        signal: Input signal [N]
        low: Low threshold (output goes to 0 when signal drops below)
        high: High threshold (output goes to 1 when signal exceeds)

    Returns:
        Binary output [N]
    """
    N = len(signal)
    output = np.zeros(N, dtype=np.float32)
    state = 0.0

    for i in range(N):
        if signal[i] > high:
            state = 1.0
        elif signal[i] < low:
            state = 0.0
        output[i] = state

    return output


# ============================================================
# Binary Slicer (replicates gr::digital::binary_slicer_fb)
# ============================================================
def binary_slicer(signal: np.ndarray) -> np.ndarray:
    """
    Slice signal at zero threshold.

    Args:
        signal: Input signal [N]

    Returns:
        Binary output [N] (0 or 1)
    """
    return (signal > 0).astype(np.float32)


# ============================================================
# Complete Spike Detection Pipeline
# ============================================================
def detect_spikes(
    iq_data: np.ndarray,
    rise_thresh: float = 10.0,
    fall_thresh: float = 20.0,
    look_ahead: int = 10,
    alpha: float = 0.001,
    th_low: float = -20.0,
    th_high: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Full spike detection pipeline (mirrors GRC flowgraph).

    Pipeline: IQ → |magnitude| → PeakDetector → Threshold → BinarySlicer

    Args:
        iq_data: Complex IQ array [N] or real array [N, 2]
        rise_thresh: Peak detector rise threshold
        fall_thresh: Peak detector fall threshold
        look_ahead: Peak detector look-ahead
        alpha: Running average smoothing
        th_low: Threshold block low value
        th_high: Threshold block high value

    Returns:
        Tuple of (binary_output, peak_locations, spike_density)
    """
    # Convert to complex if needed
    if iq_data.ndim == 2 and iq_data.shape[1] == 2:
        iq_complex = iq_data[:, 0] + 1j * iq_data[:, 1]
    else:
        iq_complex = iq_data

    # Step 1: Complex → Float (magnitude)
    amplitude = np.abs(iq_complex).astype(np.float32)

    # Step 2: Peak Detector
    peaks = peak_detector(amplitude, rise_thresh, fall_thresh, look_ahead, alpha)

    # Step 3: Threshold
    thresholded = threshold_block(peaks, th_low, th_high)

    # Step 4: Binary Slicer
    binary = binary_slicer(thresholded)

    # Spike density = fraction of samples flagged as spikes
    spike_density = np.mean(binary)

    return binary, peaks, spike_density


# ============================================================
# Spike-Based Classifier (Baseline 1)
# ============================================================
class SpikeBasedClassifier:
    """
    Traditional spike-density based classifier.

    Decision logic:
      - Compute spike density over each IQ window
      - If density < normal_thresh → Normal
      - Else classify based on spike pattern characteristics

    This is the non-ML baseline that the CNN must beat.
    """

    def __init__(
        self,
        normal_thresh: float = 0.01,
        detection_params: dict = None,
    ):
        self.normal_thresh = normal_thresh
        self.params = detection_params or {
            'rise_thresh': 10.0,
            'fall_thresh': 20.0,
            'look_ahead': 10,
            'alpha': 0.001,
        }
        self.fitted = False
        self.class_thresholds = {}

    def fit(self, windows: np.ndarray, labels: np.ndarray):
        """
        Learn spike density thresholds from training data.

        Args:
            windows: IQ windows [N, W, 2]
            labels: Class labels [N]
        """
        from collections import defaultdict
        densities_by_class = defaultdict(list)

        for i in range(len(windows)):
            _, _, density = detect_spikes(windows[i], **self.params)
            densities_by_class[labels[i]].append(density)

        # Compute per-class density statistics
        for cls, densities in densities_by_class.items():
            self.class_thresholds[cls] = {
                'mean': np.mean(densities),
                'std': np.std(densities),
                'min': np.min(densities),
                'max': np.max(densities),
            }

        self.fitted = True
        print("\nSpike density statistics per class:")
        for cls, stats in self.class_thresholds.items():
            print(f"  Class {cls}: mean={stats['mean']:.4f}, "
                  f"std={stats['std']:.4f}, "
                  f"range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """
        Classify windows based on spike density.

        Args:
            windows: IQ windows [N, W, 2]

        Returns:
            Predicted labels [N]
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before predict()")

        predictions = np.zeros(len(windows), dtype=np.int64)
        for i in range(len(windows)):
            _, _, density = detect_spikes(windows[i], **self.params)

            # Find nearest class by density
            min_dist = float('inf')
            best_class = 0
            for cls, stats in self.class_thresholds.items():
                dist = abs(density - stats['mean'])
                if dist < min_dist:
                    min_dist = dist
                    best_class = cls

            predictions[i] = best_class

        return predictions

    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """
        Compute pseudo-probabilities based on density distance.

        Args:
            windows: IQ windows [N, W, 2]

        Returns:
            Probability matrix [N, num_classes]
        """
        num_classes = len(self.class_thresholds)
        probas = np.zeros((len(windows), num_classes), dtype=np.float32)

        for i in range(len(windows)):
            _, _, density = detect_spikes(windows[i], **self.params)

            distances = np.zeros(num_classes)
            for cls, stats in self.class_thresholds.items():
                distances[cls] = abs(density - stats['mean']) / (stats['std'] + 1e-8)

            # Convert distances to probabilities (softmax of negative distances)
            neg_dist = -distances
            exp_dist = np.exp(neg_dist - np.max(neg_dist))
            probas[i] = exp_dist / np.sum(exp_dist)

        return probas


# ============================================================
# Quick Test
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)

    # Normal signal
    t = np.arange(256) / 1.92e6
    normal = (np.cos(2 * np.pi * 1000 * t) + 1j * np.sin(2 * np.pi * 1000 * t)).astype(np.complex64)
    normal += 0.1 * (np.random.randn(256) + 1j * np.random.randn(256)).astype(np.complex64)

    # Attack signal (with added tone jammer)
    attack = normal + 2.0 * np.exp(1j * 2 * np.pi * 50000 * t).astype(np.complex64)

    print("Normal signal:")
    binary_n, peaks_n, density_n = detect_spikes(normal)
    print(f"  Spike density: {density_n:.4f}")

    print("Attack signal:")
    binary_a, peaks_a, density_a = detect_spikes(attack)
    print(f"  Spike density: {density_a:.4f}")
