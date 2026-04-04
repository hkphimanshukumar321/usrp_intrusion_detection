#!/usr/bin/env python3
"""
sim_system.py — Simulates RF Multi-path Intrusion (Device-Free RF Sensing).

Simulates a Continuous Wave (CW) or QPSK transmission undergoing 
physical multi-path scattering caused by different physical intruders 
crossing the Line-of-Sight (LoS) between the TX and RX antennas.

Classes:
  0. Clear  (No Intrusion, static channel)
  1. Human  (Walking pace ~1.5 m/s, large RCS, distinct arm/leg micro-Doppler)
  2. Animal (Faster pace ~3 m/s, smaller RCS, high-frequency limb motion)
  3. Drone  (High RPM rotor micro-Doppler signatures, hovering/drifting)
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


# ============================================================
# Simulator Constants
# ============================================================
SAMP_RATE = 1.92e6          # 1.92 MSps (USRP default)
CARRIER_FREQ = 2.4e9        # 2.4 GHz ISM band
SPEED_OF_LIGHT = 3e8        # m/s
WAVELENGTH = SPEED_OF_LIGHT / CARRIER_FREQ


def QPSK_source(num_symbols: int, sps: int = 10) -> np.ndarray:
    """Generate a clean QPSK baseband signal."""
    symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], size=num_symbols)
    symbols = symbols / np.sqrt(2)
    
    # Pulse shaping (simple moving average for smoothing)
    upsampled = np.repeat(symbols, sps)
    window = np.ones(sps) / sps
    baseband = np.convolve(upsampled, window, mode='same')
    
    power = np.mean(np.abs(baseband)**2)
    return (baseband / np.sqrt(power)).astype(np.complex64)


def apply_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Apply Additive White Gaussian Noise based on target SNR."""
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    return (signal + noise).astype(np.complex64)


def apply_fading_and_micro_doppler(
    baseband: np.ndarray,
    fs: float,
    main_speed: float,          # Main body speed (m/s)
    rcs_main: float,            # Radar Cross Section of main body
    appendages: list,           # List of dicts describing limbs: {'speed_amp': float, 'rate': float, 'rcs': float}
) -> np.ndarray:
    """
    Simulate multi-path fading caused by a physical intruder crossing the LoS.
    Creates micro-Doppler signatures by modulating the phase and amplitude of the scattered multipath.
    """
    t = np.arange(len(baseband)) / fs
    
    # 1. Static Line-of-Sight (LoS) path
    los_path = baseband * 1.0  # Dominant path
    
    # 2. Dynamic scattering path (The Intruder)
    # The Doppler shift is f_d = (2 * v * f_c) / c * cos(theta). 
    # For a crossing target, the radial velocity changes, but we'll approximate 
    # specific micro-Doppler variations as sinusoidal phase modulations.
    
    main_doppler = (2 * main_speed) / WAVELENGTH
    base_phase = 2 * np.pi * main_doppler * t
    scattered_path = rcs_main * np.exp(1j * base_phase)
    
    # Add micro-Doppler from limbs/rotors
    for app in appendages:
        # Appendage velocity oscillates around the main body velocity
        # The phase contribution integral of sinusoidal velocity is a cosine
        d_phase = (app['speed_amp'] / app['rate']) * np.cos(2 * np.pi * app['rate'] * t)
        app_phase = base_phase + (2 * np.pi / WAVELENGTH) * d_phase
        scattered_path += app['rcs'] * np.exp(1j * app_phase)
    
    # The received signal is the superposition of LoS and scattered paths
    rx_signal = los_path * (1 + scattered_path)
    
    # Normalize power to preserve unit variance before adding channel noise
    rx_signal = rx_signal / np.sqrt(np.mean(np.abs(rx_signal)**2))
    return rx_signal.astype(np.complex64)


def simulate_class(
    class_id: int, 
    num_samples: int, 
    fs: float = SAMP_RATE, 
    snr_db: float = 20.0
) -> np.ndarray:
    """Generate the received baseband signal for a specific intruder class."""
    
    # 1. Generate base QPSK transmission
    sps = 10
    num_symbols = (num_samples + sps - 1) // sps
    baseband = QPSK_source(num_symbols, sps=sps)[:num_samples]
    
    # 2. Apply Physical Intrusion Multi-path
    if class_id == 0:
        # CLEAR (No intrusion)
        # Just small, slow environmental variations (e.g. slight breeze on furniture)
        signal = apply_fading_and_micro_doppler(
            baseband, fs, main_speed=0.01, rcs_main=0.02, appendages=[]
        )
        
    elif class_id == 1:
        # HUMAN
        # Walking pace: ~1.5 m/s. High RCS.
        # Arms/Legs swinging: ~2 Hz rate, speed amplitude ~1.0 m/s
        signal = apply_fading_and_micro_doppler(
            baseband, fs, 
            main_speed=1.5, 
            rcs_main=0.4, 
            appendages=[
                {'speed_amp': 1.0, 'rate': 2.0, 'rcs': 0.15}, # Legs
                {'speed_amp': 0.8, 'rate': 2.0, 'rcs': 0.10}, # Arms
            ]
        )
        
    elif class_id == 2:
        # ANIMAL (e.g. Dog)
        # Faster trot: ~3 m/s. Smaller RCS.
        # High frequency gait: ~4-5 Hz rate.
        signal = apply_fading_and_micro_doppler(
            baseband, fs, 
            main_speed=3.0, 
            rcs_main=0.15, 
            appendages=[
                {'speed_amp': 1.5, 'rate': 4.5, 'rcs': 0.08}, # Front legs
                {'speed_amp': 1.5, 'rate': 4.5, 'rcs': 0.08}, # Back legs
            ]
        )
        
    elif class_id == 3:
        # DRONE / MACHINE
        # Slow drifting: ~0.5 m/s. Medium RCS.
        # Rotors: High RPM spinning creates very high-frequency micro-Doppler (~50 Hz blade pass).
        signal = apply_fading_and_micro_doppler(
            baseband, fs, 
            main_speed=0.5, 
            rcs_main=0.25, 
            appendages=[
                {'speed_amp': 15.0, 'rate': 50.0, 'rcs': 0.1}, # Quadcopter blades
            ]
        )
    else:
        raise ValueError("Invalid class ID")

    # 3. Apply AWGN receiver noise
    rx_noisy = apply_awgn(signal, snr_db)
    
    return rx_noisy


def generate_dataset(
    output_dir: str = "dataset/simulated",
    duration_sec: float = 10.0,
    snr_db: float = 20.0,
):
    """Generate the full benchmark dataset for all 4 physical intrusion classes."""
    os.makedirs(output_dir, exist_ok=True)
    num_samples = int(SAMP_RATE * duration_sec)
    
    classes = {
        0: "clear.dat",
        1: "human.dat",
        2: "animal.dat",
        3: "drone.dat",
    }
    
    print(f"\n{'='*50}")
    print(f"Generating Physical Intrusion Dataset")
    print(f"  Duration:     {duration_sec} s per class")
    print(f"  Sample Rate:  {SAMP_RATE/1e6:.2f} MSps")
    print(f"  SNR:          {snr_db} dB")
    print(f"{'='*50}\n")
    
    for cls_id, filename in classes.items():
        filepath = os.path.join(output_dir, filename)
        print(f"[{cls_id+1}/{len(classes)}] Generating {filename.split('.')[0].upper()} profile...")
        
        signal = simulate_class(cls_id, num_samples, SAMP_RATE, snr_db)
        
        # Save as complex64 (GNU Radio format)
        signal.astype(np.complex64).tofile(filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  ✓ Saved to {filepath} ({size_mb:.1f} MB)")
        
    print(f"\n✅ Physical Intrusion Dataset generated in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Device-Free RF Intrusion (Burglar Alarm)")
    parser.add_argument("--output_dir", type=str, default="dataset/simulated")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per class")
    parser.add_argument("--snr_db", type=float, default=20.0, help="Channel SNR")
    args = parser.parse_args()

    generate_dataset(args.output_dir, args.duration, args.snr_db)
