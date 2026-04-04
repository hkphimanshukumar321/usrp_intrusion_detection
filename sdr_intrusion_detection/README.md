# Device-Free RF Physical Intrusion Sensing System

> **"From Spikes to CNNs: Dual-Representation Deep Learning for Biological and Drone Intrusion Classification via RF Micro-Doppler Scattering"**

## Overview
This project constitutes a complete end-to-end framework for a **Device-Free Physical Intrusion Detector** (A Burglar Alarm using RF). By utilizing Software Defined Radios (SDR), the system propagates a continuous RF wave. When humans, animals, or drones cross the invisible Line-of-Sight (LoS) path, their physical forms scatter the transmission.

Instead of a simple peak-detector that alarms indiscriminately, this system parses the `I + jQ` fluctuations and applies a Deep Learning pipeline (including a custom PyTorch DenseNet) to classify whether the intruder is a Human (walking), an Animal (trotting), or a Drone (spinning rotors).

---

## Remote Server Deployment (Headless Execution)

The entire `src/` stack is fully optimized for remote Linux/Windows SSH servers equipped with NVIDIA Graphics Cards.

### 1. Installation
1. Clone this repository directly onto the server instance. Note that the `.gitignore` safely strips `/dataset/` and `/results/` raw binary files, meaning the clone operation will be instant.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Full Headless Execution
Execute the master ablation script. This will use all available CPU cores (`num_workers=os.cpu_count()`) for batch processing the raw datasets, while pinning calculations to the primary CUDA GPU.
```bash
# This will:
# 1. Generate 10 seconds of synthetic data per class (Clear, Human, Animal, Drone)
# 2. Extract statistical features and 2D Micro-Doppler STFT Spectrograms.
# 3. Train all architectures simultaneously via 5-Fold Stratified Cross-Validation (including Custom DenseNet).
# 4. Generate Confusion Matrices and ROC Curves.

python -m src.run_ablation --duration 10.0
```

> **Note:** Training states, best fold epochs, and `.pth` checkpoint outputs are automatically saved to `results/models/` but ignored from Git tracking to preserve repo size.

---

## Hardware Setup & Data Collection (GNURadio)

Instead of simulating on the server, you may capture real datasets using a **USRP B210**:

1. **Antenna Placement:** Place the TX and RX antennas on opposite sides of a hallway or doorway.
2. **Run the GRC Collector:** Boot up `sdr_flowgraphs/usrp_data_collector.grc` in GNURadio.
3. **Capture Dataset (.dat):** Assign `Target Class Name` to `"human"`. Have a subject cross the path. Repeat for `"drone"` and `"animal"`.
4. **Train your network via SSH:** Push those recorded datasets to the server and execute:
```bash
# Skips the data generation logic, using your uploaded .dat recordings directly!
python -m src.run_ablation --skip_data
```

## Notebooks Workflow
Check `notebooks/pipeline_walkthrough.ipynb` for a fully interactive tutorial demonstrating:
- **Case 1:** Executing the pipeline completely artificially in Python.
- **Case 2:** Using GNURadio flowgraphs to harvest and analyze real hardware data.

## Models Evaluated
- **Custom DenseNet:** User-provided ultra-lightweight block-compression architecture.
- **CNN2D_Spec:** Deep 2D Convolutions mapping micro-Doppler time-varying spectra.
- **Dual_Branch_Fusion:** Advanced Squeeze-and-Excitation fusion combining 1D (Time Fading) + 2D (Frequency Rotor) inputs.
- Non-ML Baselines (Spike Detection thresholds, Scikit-learn SVMs).
