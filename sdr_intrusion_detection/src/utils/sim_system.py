#!/usr/bin/env python3
"""
sim_system.py - Scenario-based RF intrusion simulator with richer channel realism.

This generator is intended for study-quality synthetic pretraining, not as a claim
of deployment-ready realism. It improves over the old single-trace-per-class setup by:
  - randomizing scenario physics per sample
  - varying SNR, clutter, multipath, and crossing geometry
  - injecting hardware impairments (CFO, phase noise, IQ imbalance, DC offset,
    AGC drift, sample-clock drift)
  - saving scenario metadata so downstream code can perform scenario-level splits
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


# ============================================================
# Simulator Constants
# ============================================================
SAMP_RATE = 1.92e6
CARRIER_FREQ = 2.4e9
SPEED_OF_LIGHT = 3e8
WAVELENGTH = SPEED_OF_LIGHT / CARRIER_FREQ
CLASS_FILES = {
    0: "clear.dat",
    1: "human.dat",
    2: "animal.dat",
    3: "drone.dat",
}
CLASS_NAMES = {
    0: "clear",
    1: "human",
    2: "animal",
    3: "drone",
}


def normalize_complex(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    power = np.mean(np.abs(signal) ** 2)
    return (signal / np.sqrt(power + eps)).astype(np.complex64)


def qpsk_source(num_symbols: int, sps: int, rng: np.random.Generator) -> np.ndarray:
    """Generate shaped QPSK baseband."""
    symbols = rng.choice(
        np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64),
        size=num_symbols,
    )
    symbols = symbols / np.sqrt(2)
    upsampled = np.repeat(symbols, sps)
    pulse = np.hanning(max(4, 2 * sps + 1))
    pulse = pulse / np.sum(pulse)
    shaped = np.convolve(upsampled, pulse, mode="same")
    return normalize_complex(shaped.astype(np.complex64))


def apply_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(len(signal)) + 1j * rng.standard_normal(len(signal))
    )
    return (signal + noise).astype(np.complex64)


def apply_time_warp(
    signal: np.ndarray,
    ppm: float,
    drift_strength: float,
) -> np.ndarray:
    """Approximate sampling-clock drift with a slow non-linear time warp."""
    n = len(signal)
    if n <= 2 or (abs(ppm) < 1e-12 and abs(drift_strength) < 1e-12):
        return signal.astype(np.complex64)

    base = np.arange(n, dtype=np.float64)
    linear = base * (1.0 + ppm * 1e-6)
    curve = drift_strength * (base / max(n - 1, 1)) ** 2 * n
    warped = np.clip(linear + curve, 0, n - 1)

    real = np.interp(base, warped, signal.real)
    imag = np.interp(base, warped, signal.imag)
    return (real + 1j * imag).astype(np.complex64)


def apply_rf_impairments(
    signal: np.ndarray,
    fs: float,
    impairments: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject practical SDR/RF front-end effects."""
    signal = apply_time_warp(
        signal,
        impairments["sample_clock_ppm"],
        impairments["clock_curve_strength"],
    )

    n = len(signal)
    t = np.arange(n, dtype=np.float64) / fs

    phase_noise = np.cumsum(
        rng.normal(0.0, impairments["phase_noise_std"], size=n)
    )
    signal = signal * np.exp(1j * (2 * np.pi * impairments["cfo_hz"] * t + phase_noise))

    i = signal.real
    q = signal.imag
    gain_i = 10 ** (impairments["iq_gain_db"] / 20.0)
    gain_q = 10 ** (-impairments["iq_gain_db"] / 20.0)
    phase_skew = np.deg2rad(impairments["iq_phase_deg"])
    i_new = gain_i * i
    q_new = gain_q * (q * np.cos(phase_skew) + i * np.sin(phase_skew))
    signal = i_new + 1j * q_new

    agc_wave = 1.0 + impairments["agc_depth"] * np.sin(
        2 * np.pi * impairments["agc_rate_hz"] * t + impairments["agc_phase"]
    )
    agc_rw = np.cumsum(rng.normal(0.0, impairments["agc_rw_std"], size=n))
    signal = signal * np.clip(agc_wave + agc_rw, 0.7, 1.3)

    dc = impairments["dc_i"] + 1j * impairments["dc_q"]
    return (signal + dc).astype(np.complex64)


def add_interference(
    signal: np.ndarray,
    fs: float,
    interference_cfg: Dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Add narrowband interferers, jammer tones, or weak chirps."""
    n = len(signal)
    t = np.arange(n, dtype=np.float64) / fs
    out = signal.astype(np.complex64)

    for tone in interference_cfg["tones"]:
        out = out + tone["amp"] * np.exp(1j * (2 * np.pi * tone["freq_hz"] * t + tone["phase"]))

    if interference_cfg["chirp_amp"] > 0:
        f0 = interference_cfg["chirp_f0_hz"]
        f1 = interference_cfg["chirp_f1_hz"]
        chirp_phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / max(t[-1], 1e-9) * t ** 2)
        out = out + interference_cfg["chirp_amp"] * np.exp(1j * chirp_phase)

    return out.astype(np.complex64)


def build_impairment_config(rng: np.random.Generator) -> Dict[str, float]:
    return {
        "cfo_hz": float(rng.uniform(-3_000.0, 3_000.0)),
        "phase_noise_std": float(rng.uniform(1e-5, 8e-4)),
        "iq_gain_db": float(rng.uniform(-1.5, 1.5)),
        "iq_phase_deg": float(rng.uniform(-6.0, 6.0)),
        "dc_i": float(rng.uniform(-0.02, 0.02)),
        "dc_q": float(rng.uniform(-0.02, 0.02)),
        "agc_depth": float(rng.uniform(0.01, 0.08)),
        "agc_rate_hz": float(rng.uniform(0.2, 2.5)),
        "agc_phase": float(rng.uniform(0.0, 2 * np.pi)),
        "agc_rw_std": float(rng.uniform(1e-5, 6e-5)),
        "sample_clock_ppm": float(rng.uniform(-20.0, 20.0)),
        "clock_curve_strength": float(rng.uniform(-5e-4, 5e-4)),
    }


def build_interference_config(rng: np.random.Generator) -> Dict[str, float]:
    tones = []
    if rng.random() < 0.65:
        for _ in range(int(rng.integers(1, 3))):
            tones.append({
                "freq_hz": float(rng.uniform(-180_000.0, 180_000.0)),
                "amp": float(rng.uniform(0.01, 0.08)),
                "phase": float(rng.uniform(0.0, 2 * np.pi)),
            })

    chirp_amp = float(rng.uniform(0.01, 0.05)) if rng.random() < 0.2 else 0.0
    return {
        "tones": tones,
        "chirp_amp": chirp_amp,
        "chirp_f0_hz": float(rng.uniform(-120_000.0, -20_000.0)),
        "chirp_f1_hz": float(rng.uniform(20_000.0, 120_000.0)),
    }


def sample_appendages(
    class_id: int,
    main_speed: float,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    if class_id == 0:
        return []
    if class_id == 1:
        count = int(rng.integers(2, 5))
        rate_range = (1.2, 2.8)
        speed_amp_range = (0.3, 1.3)
        rcs_range = (0.03, 0.14)
    elif class_id == 2:
        count = int(rng.integers(2, 5))
        rate_range = (3.0, 7.5)
        speed_amp_range = (0.8, 2.2)
        rcs_range = (0.02, 0.08)
    else:
        count = int(rng.integers(2, 5))
        rate_range = (35.0, 120.0)
        speed_amp_range = (6.0, 18.0)
        rcs_range = (0.02, 0.09)

    appendages = []
    for _ in range(count):
        appendages.append({
            "speed_amp": float(rng.uniform(*speed_amp_range)),
            "rate": float(rng.uniform(*rate_range)),
            "rcs": float(rng.uniform(*rcs_range)),
            "phase": float(rng.uniform(0.0, 2 * np.pi)),
            "delay_s": float(rng.uniform(0.0, 2.5e-6)),
        })
    return appendages


def sample_class_profile(class_id: int, rng: np.random.Generator) -> Dict[str, float]:
    if class_id == 0:
        return {
            "main_speed": float(rng.uniform(0.0, 0.08)),
            "rcs_main": float(rng.uniform(0.005, 0.03)),
            "crossing_angle_deg": float(rng.uniform(40.0, 90.0)),
            "num_clutter_paths": int(rng.integers(3, 7)),
            "clutter_doppler_hz": float(rng.uniform(0.1, 4.0)),
            "appendages": [],
            "snr_db": float(rng.uniform(8.0, 30.0)),
        }
    if class_id == 1:
        main_speed = float(rng.uniform(0.8, 2.2))
        return {
            "main_speed": main_speed,
            "rcs_main": float(rng.uniform(0.20, 0.50)),
            "crossing_angle_deg": float(rng.uniform(20.0, 80.0)),
            "num_clutter_paths": int(rng.integers(2, 6)),
            "clutter_doppler_hz": float(rng.uniform(0.2, 5.0)),
            "appendages": sample_appendages(class_id, main_speed, rng),
            "snr_db": float(rng.uniform(4.0, 24.0)),
        }
    if class_id == 2:
        main_speed = float(rng.uniform(1.6, 4.5))
        return {
            "main_speed": main_speed,
            "rcs_main": float(rng.uniform(0.08, 0.22)),
            "crossing_angle_deg": float(rng.uniform(20.0, 85.0)),
            "num_clutter_paths": int(rng.integers(2, 6)),
            "clutter_doppler_hz": float(rng.uniform(0.4, 7.0)),
            "appendages": sample_appendages(class_id, main_speed, rng),
            "snr_db": float(rng.uniform(2.0, 22.0)),
        }

    main_speed = float(rng.uniform(0.0, 1.2))
    return {
        "main_speed": main_speed,
        "rcs_main": float(rng.uniform(0.10, 0.30)),
        "crossing_angle_deg": float(rng.uniform(10.0, 75.0)),
        "num_clutter_paths": int(rng.integers(2, 6)),
        "clutter_doppler_hz": float(rng.uniform(0.2, 4.0)),
        "appendages": sample_appendages(class_id, main_speed, rng),
        "snr_db": float(rng.uniform(0.0, 20.0)),
    }


def delayed(signal: np.ndarray, delay_samples: int) -> np.ndarray:
    if delay_samples <= 0:
        return signal
    out = np.zeros_like(signal)
    out[delay_samples:] = signal[:-delay_samples]
    return out


def simulate_scenario_signal(
    class_id: int,
    num_samples: int,
    fs: float,
    scenario_cfg: Dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one randomized physical scenario."""
    sps = int(rng.choice([6, 8, 10, 12]))
    num_symbols = (num_samples + sps - 1) // sps
    baseband = qpsk_source(num_symbols, sps=sps, rng=rng)[:num_samples]
    t = np.arange(num_samples, dtype=np.float64) / fs

    los_amp = float(rng.uniform(0.8, 1.2))
    los_phase = float(rng.uniform(0.0, 2 * np.pi))
    rx = los_amp * np.exp(1j * los_phase) * baseband

    for _ in range(scenario_cfg["num_clutter_paths"]):
        delay_samples = int(rng.integers(0, 10))
        gain = float(rng.uniform(0.02, 0.18))
        phase = float(rng.uniform(0.0, 2 * np.pi))
        clutter_fd = float(rng.uniform(-scenario_cfg["clutter_doppler_hz"], scenario_cfg["clutter_doppler_hz"]))
        clutter_env = 1.0 + 0.15 * np.sin(
            2 * np.pi * float(rng.uniform(0.1, 1.5)) * t + float(rng.uniform(0.0, 2 * np.pi))
        )
        rx += gain * clutter_env * delayed(baseband, delay_samples) * np.exp(1j * (phase + 2 * np.pi * clutter_fd * t))

    crossing_center = float(rng.uniform(0.25, 0.75)) * t[-1]
    crossing_width = max(float(rng.uniform(0.08, 0.22)) * t[-1], 1e-3)
    target_env = np.exp(-0.5 * ((t - crossing_center) / crossing_width) ** 2)
    target_env = target_env / (np.max(target_env) + 1e-8)

    radial_speed = scenario_cfg["main_speed"] * np.cos(np.deg2rad(scenario_cfg["crossing_angle_deg"]))
    body_fd = (2.0 * radial_speed) / WAVELENGTH
    body_phase = 2 * np.pi * body_fd * t + float(rng.uniform(0.0, 2 * np.pi))
    target_delay = int(rng.integers(1, 14))
    rx += (
        scenario_cfg["rcs_main"] * target_env *
        delayed(baseband, target_delay) *
        np.exp(1j * body_phase)
    )

    for app in scenario_cfg["appendages"]:
        phase_mod = (2 * np.pi / WAVELENGTH) * (
            app["speed_amp"] / max(app["rate"], 1e-3)
        ) * np.cos(2 * np.pi * app["rate"] * t + app["phase"])
        app_delay = int(max(0, round(app["delay_s"] * fs)))
        rx += (
            app["rcs"] *
            target_env *
            delayed(baseband, app_delay) *
            np.exp(1j * (body_phase + phase_mod))
        )

    rx = normalize_complex(rx)
    rx = apply_rf_impairments(rx, fs, scenario_cfg["impairments"], rng)
    rx = add_interference(rx, fs, scenario_cfg["interference"], rng)
    rx = normalize_complex(rx)
    rx = apply_awgn(rx, scenario_cfg["snr_db"], rng)
    return rx.astype(np.complex64)


def split_duration_across_scenarios(
    total_samples: int,
    scenarios_per_class: int,
    rng: np.random.Generator,
) -> List[int]:
    min_samples = min(2048, max(512, total_samples // max(2 * scenarios_per_class, 1)))
    weights = rng.dirichlet(np.full(scenarios_per_class, 2.0))
    raw = np.maximum((weights * total_samples).astype(int), min_samples)
    current = int(raw.sum())
    while current > total_samples:
        idx = int(np.argmax(raw))
        if raw[idx] <= min_samples:
            break
        raw[idx] -= 1
        current -= 1
    while current < total_samples:
        idx = int(np.argmin(raw))
        raw[idx] += 1
        current += 1
    return raw.tolist()


def build_scenario_config(
    class_id: int,
    scenario_id: str,
    num_samples: int,
    rng: np.random.Generator,
) -> Dict:
    profile = sample_class_profile(class_id, rng)
    profile["scenario_id"] = scenario_id
    profile["num_samples"] = int(num_samples)
    profile["duration_sec"] = float(num_samples / SAMP_RATE)
    profile["impairments"] = build_impairment_config(rng)
    profile["interference"] = build_interference_config(rng)
    return profile


def generate_dataset(
    output_dir: str = "dataset/simulated",
    duration_sec: float = 10.0,
    snr_db: float = 20.0,
    seed: int = 42,
    scenarios_per_class: int = 12,
):
    """Generate a scenario-diverse benchmark dataset for all classes."""
    del snr_db  # kept for backwards CLI compatibility; scenario SNR is randomized now.

    os.makedirs(output_dir, exist_ok=True)
    total_samples_per_class = int(SAMP_RATE * duration_sec)
    master_rng = np.random.default_rng(seed)
    manifest = {
        "version": 2,
        "seed": seed,
        "sample_rate": SAMP_RATE,
        "carrier_freq": CARRIER_FREQ,
        "duration_sec_per_class": duration_sec,
        "scenarios_per_class": scenarios_per_class,
        "notes": [
            "Synthetic pretraining dataset with randomized scenario physics and RF impairments.",
            "Use scenario-level cross-validation to avoid optimistic leakage across near-duplicate windows.",
            "Real SDR validation is still required for sim-to-real claims.",
        ],
        "classes": {},
    }

    print(f"\n{'='*60}")
    print("Generating Scenario-Diverse RF Intrusion Dataset")
    print(f"  Duration:          {duration_sec:.2f} s per class")
    print(f"  Sample Rate:       {SAMP_RATE/1e6:.2f} MSps")
    print(f"  Seed:              {seed}")
    print(f"  Scenarios/Class:   {scenarios_per_class}")
    print(f"{'='*60}\n")

    for class_id, filename in CLASS_FILES.items():
        class_name = CLASS_NAMES[class_id]
        filepath = os.path.join(output_dir, filename)
        print(f"[{class_id+1}/{len(CLASS_FILES)}] Generating {class_name.upper()} scenarios...")

        scenario_lengths = split_duration_across_scenarios(
            total_samples_per_class,
            scenarios_per_class,
            master_rng,
        )
        class_signal_parts = []
        class_scenarios = []
        start_sample = 0

        for scenario_idx, num_samples in enumerate(tqdm(scenario_lengths, desc=f"  {class_name} scenarios", leave=False)):
            scenario_seed = int(master_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(scenario_seed)
            scenario_id = f"{class_name}_scenario_{scenario_idx:03d}"
            scenario_cfg = build_scenario_config(class_id, scenario_id, num_samples, rng)
            signal = simulate_scenario_signal(class_id, num_samples, SAMP_RATE, scenario_cfg, rng)

            end_sample = start_sample + num_samples
            class_signal_parts.append(signal)
            class_scenarios.append({
                "scenario_id": scenario_id,
                "scenario_seed": scenario_seed,
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "config": scenario_cfg,
            })
            start_sample = end_sample

        class_signal = np.concatenate(class_signal_parts).astype(np.complex64)
        class_signal.tofile(filepath)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"  Saved to {filepath} ({size_mb:.1f} MB)")

        manifest["classes"][filename] = {
            "class_id": class_id,
            "class_name": class_name,
            "num_samples": int(len(class_signal)),
            "num_scenarios": len(class_scenarios),
            "scenarios": class_scenarios,
        }

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nDataset manifest saved to {manifest_path}")
    print(f"Scenario-diverse RF intrusion dataset generated in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate device-free RF intrusion scenarios")
    parser.add_argument("--output_dir", type=str, default="dataset/simulated")
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds per class")
    parser.add_argument("--snr_db", type=float, default=20.0, help="Deprecated compatibility flag")
    parser.add_argument("--seed", type=int, default=42, help="Dataset generation seed")
    parser.add_argument(
        "--scenarios_per_class",
        type=int,
        default=12,
        help="Number of randomized scenarios to generate per class",
    )
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        duration_sec=args.duration,
        snr_db=args.snr_db,
        seed=args.seed,
        scenarios_per_class=args.scenarios_per_class,
    )
