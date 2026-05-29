"""
build_dataset.py -- Unified RF Intrusion Detection Dataset Builder
=================================================================
Parses 3 raw datasets (Zenodo Radar, DroneRF USRP, Jamming Spectral Scans)
and generates a unified 6-class dataset of 224x224 spectrogram images
with train/val/test splits (70/15/15).

IEEE-Standard Classes:
  1. Machine          -- Mechanical/autonomous intrusions (drones, robots, vehicles)
  2. Human            -- Human intrusions (walking, running)
  3. Wildlife         -- Non-human biological intrusions (birds, animals, fauna)
  4. Broadband_Jam    -- Broadband RF interference (gaussian noise jamming)
  5. Narrowband_Jam   -- Narrowband RF interference (single-tone jamming)
  6. Benign           -- No-threat background (normal RF environment)

Output structure:
  unified_dataset/
    train/
      Machine/ Human/ Wildlife/ Broadband_Jam/ Narrowband_Jam/ Benign/
    val/
      Machine/ Human/ Wildlife/ Broadband_Jam/ Narrowband_Jam/ Benign/
    test/
      Machine/ Human/ Wildlife/ Broadband_Jam/ Narrowband_Jam/ Benign/

Usage:
  python build_dataset.py
"""

import os
import sys
import glob
import shutil
import subprocess
import numpy as np
from scipy import signal as sig
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "unified_dataset")
STAGING_DIR = os.path.join(BASE_DIR, "_staging")  # Temp before split
IMG_SIZE = 224

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# IEEE-standard class names
CLASSES = ['Machine', 'Human', 'Wildlife', 'Broadband_Jam', 'Narrowband_Jam', 'Benign']

# Dataset paths
RADAR_NPY = os.path.join(RAW_DIR, "5845259", "data_SAAB_SIRS_77GHz_FMCW.npy")
DRONERF_DIR = os.path.join(RAW_DIR, "f4c2b4n755-1", "DroneRF")
JAMMING_DIR = os.path.join(RAW_DIR, "archive (2)")

# Class mapping for Radar dataset
RADAR_CLASS_MAP = {
    'D1': 'Machine', 'D2': 'Machine', 'D3': 'Machine',
    'D4': 'Machine', 'D5': 'Machine', 'D6': 'Machine',
    'human_walk': 'Human', 'human_run': 'Human',
    'seagull': 'Wildlife', 'pigeon': 'Wildlife', 'raven': 'Wildlife',
    'black-headed gull': 'Wildlife', 'seagull and black-headed gull': 'Wildlife',
    'heron': 'Wildlife',
    'CR': None,  # Skip corner reflector
}

# DroneRF class mapping
DRONERF_CLASS_MAP = {
    '00000': 'Benign',
    '10100': 'Machine', '10101': 'Machine',
    '10110': 'Machine', '10111': 'Machine',
}

TARGET_PER_CLASS = 4000


def ensure_dirs():
    """Create staging and output directory structures."""
    for cls in CLASSES:
        os.makedirs(os.path.join(STAGING_DIR, cls), exist_ok=True)
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
    print("[OK] Directory structures created")
    print(f"     Staging: {STAGING_DIR}")
    print(f"     Output:  {OUTPUT_DIR}")


def spectrogram_to_image(spec_data, output_path):
    """Convert a 2D spectrogram array to a normalized 224x224 grayscale PNG."""
    with np.errstate(divide='ignore', invalid='ignore'):
        spec_db = 10 * np.log10(np.abs(spec_data) + 1e-12)

    smin, smax = spec_db.min(), spec_db.max()
    if smax - smin < 1e-6:
        return False
    spec_norm = ((spec_db - smin) / (smax - smin) * 255).astype(np.uint8)

    img = Image.fromarray(spec_norm, mode='L')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img.save(output_path)
    return True


# --- PHASE 1: RADAR DATASET -------------------------------------------------

def process_radar():
    """Parse the Zenodo 77GHz FMCW Radar .npy file."""
    print("\n" + "=" * 60)
    print("PHASE 1: Processing Zenodo FMCW Radar Dataset")
    print("=" * 60)

    if not os.path.exists(RADAR_NPY):
        print(f"[FAIL] Radar .npy not found at: {RADAR_NPY}")
        return {}

    print("[...] Loading radar .npy file (1.55 GB, please wait)...")
    data = np.load(RADAR_NPY, allow_pickle=True)
    print(f"[OK] Loaded: {data.shape[0]} measurements, {data.shape[1]} columns")

    counts = {}
    sample_idx = 0

    for row_idx in range(data.shape[0]):
        label_str = str(data[row_idx, 0]).strip()
        cls = RADAR_CLASS_MAP.get(label_str)

        if cls is None:
            continue

        segment_matrix = data[row_idx, 1]

        if segment_matrix is None or not hasattr(segment_matrix, 'shape'):
            continue

        n_segments = segment_matrix.shape[1] if len(segment_matrix.shape) > 1 else 1

        for seg_idx in range(n_segments):
            if len(segment_matrix.shape) > 1:
                segment = segment_matrix[:, seg_idx]
            else:
                segment = segment_matrix

            if len(segment) != 1280:
                continue
            range_azimuth = segment.reshape(5, 256)

            # Range integration then STFT for micro-Doppler
            slow_time = np.sum(range_azimuth, axis=0)

            f, t, Sxx = sig.spectrogram(
                slow_time,
                fs=17000,       # PRF = 17 kHz
                nperseg=64,
                noverlap=48,
                nfft=128,
                return_onesided=False
            )

            if Sxx.size < 16:
                continue

            out_path = os.path.join(STAGING_DIR, cls, f"radar_{label_str}_{sample_idx:06d}.png")
            if spectrogram_to_image(Sxx, out_path):
                counts[cls] = counts.get(cls, 0) + 1
                sample_idx += 1

            if sample_idx % 5000 == 0 and sample_idx > 0:
                print(f"  [{sample_idx} spectrograms generated...]")

    print(f"[OK] Radar processing complete. Counts: {counts}")
    return counts


# --- PHASE 2: DRONERF DATASET -----------------------------------------------

def extract_rar_files():
    """Extract .rar files in DroneRF directory using WinRAR, 7z, or rarfile."""
    rar_files = []
    for subdir in os.listdir(DRONERF_DIR):
        subdir_path = os.path.join(DRONERF_DIR, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                if f.endswith('.rar'):
                    rar_files.append(os.path.join(subdir_path, f))

    if not rar_files:
        print("[OK] No .rar files found (already extracted or not present).")
        return

    print(f"[...] Found {len(rar_files)} .rar files to extract.")

    for rar_path in rar_files:
        extract_dir = os.path.dirname(rar_path)
        basename = os.path.splitext(os.path.basename(rar_path))[0]
        target_dir = os.path.join(extract_dir, basename)

        if os.path.exists(target_dir) and os.listdir(target_dir):
            print(f"  [skip] Already extracted: {basename}")
            continue

        os.makedirs(target_dir, exist_ok=True)
        extracted = False

        # Method 1: WinRAR (confirmed available on this system)
        winrar = r"C:\Program Files\WinRAR\UnRAR.exe"
        if os.path.exists(winrar):
            try:
                result = subprocess.run(
                    [winrar, "x", "-y", rar_path, target_dir + "\\"],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    print(f"  [OK] Extracted (WinRAR): {basename}")
                    extracted = True
            except Exception:
                pass

        # Method 2: 7z fallback
        if not extracted:
            for sevenz in ["7z", r"C:\Program Files\7-Zip\7z.exe"]:
                try:
                    result = subprocess.run(
                        [sevenz, "x", rar_path, f"-o{target_dir}", "-y"],
                        capture_output=True, text=True, timeout=300
                    )
                    if result.returncode == 0:
                        print(f"  [OK] Extracted (7z): {basename}")
                        extracted = True
                        break
                except Exception:
                    pass

        # Method 3: rarfile module
        if not extracted:
            try:
                import rarfile
                rf = rarfile.RarFile(rar_path)
                rf.extractall(target_dir)
                rf.close()
                print(f"  [OK] Extracted (rarfile): {basename}")
                extracted = True
            except Exception:
                pass

        if not extracted:
            print(f"  [FAIL] Could not extract: {basename}")
            print(f"         Please extract manually: {rar_path}")


def process_dronerf():
    """Parse DroneRF raw RF I/Q data files (USRP B210, 2.4/5.8 GHz)."""
    print("\n" + "=" * 60)
    print("PHASE 2: Processing DroneRF (USRP) Dataset")
    print("=" * 60)

    if not os.path.exists(DRONERF_DIR):
        print(f"[FAIL] DroneRF directory not found: {DRONERF_DIR}")
        return {}

    extract_rar_files()

    counts = {}
    sample_idx = 0

    for subdir_name in os.listdir(DRONERF_DIR):
        subdir_path = os.path.join(DRONERF_DIR, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        for root, dirs, files in os.walk(subdir_path):
            for fname in files:
                fpath = os.path.join(root, fname)

                if fname.endswith('.rar'):
                    continue

                cls = None
                for code, mapped_cls in DRONERF_CLASS_MAP.items():
                    if code in fname or code in root:
                        cls = mapped_cls
                        break

                if cls is None:
                    if 'background' in root.lower():
                        cls = 'Benign'
                    elif any(d in root.lower() for d in ['ar drone', 'bepop', 'bebop', 'phantom']):
                        cls = 'Machine'
                    else:
                        continue

                try:
                    file_size = os.path.getsize(fpath)
                    if file_size < 1024:
                        continue

                    raw = np.fromfile(fpath, dtype=np.float32)
                    if len(raw) < 512:
                        continue

                    if len(raw) % 2 == 0:
                        iq = raw[::2] + 1j * raw[1::2]
                    else:
                        iq = raw[:-1:2] + 1j * raw[1::2]

                    segment_len = 8192
                    n_chunks = min(len(iq) // segment_len, 500)

                    for chunk_idx in range(n_chunks):
                        chunk = iq[chunk_idx * segment_len : (chunk_idx + 1) * segment_len]

                        f, t, Sxx = sig.spectrogram(
                            chunk,
                            fs=20e6,
                            nperseg=256,
                            noverlap=192,
                            nfft=512,
                            return_onesided=False
                        )

                        if Sxx.size < 16:
                            continue

                        out_path = os.path.join(STAGING_DIR, cls, f"dronerf_{sample_idx:06d}.png")
                        if spectrogram_to_image(Sxx, out_path):
                            counts[cls] = counts.get(cls, 0) + 1
                            sample_idx += 1

                        if sample_idx % 1000 == 0 and sample_idx > 0:
                            print(f"  [{sample_idx} spectrograms generated...]")

                except Exception:
                    continue

    print(f"[OK] DroneRF processing complete. Counts: {counts}")
    return counts


# --- PHASE 3: JAMMING DATASET -----------------------------------------------

def process_jamming():
    """Parse the RF Jamming Spectral Scan CSV files."""
    print("\n" + "=" * 60)
    print("PHASE 3: Processing RF Jamming Spectral Scan Dataset")
    print("=" * 60)

    if not os.path.exists(JAMMING_DIR):
        print(f"[FAIL] Jamming directory not found: {JAMMING_DIR}")
        return {}

    counts = {}
    sample_idx = 0

    scan_configs = [
        ('active_scan', 'malicious', 'gaussian_noise', 'Broadband_Jam'),
        ('active_scan', 'malicious', 'singletone', 'Narrowband_Jam'),
        ('active_scan', 'benign', 'background', 'Benign'),
        ('active_scan', 'benign', 'floor', 'Benign'),
        ('passive_scan', 'malicious', 'gaussian_noise', 'Broadband_Jam'),
        ('passive_scan', 'benign', 'background', 'Benign'),
        ('passive_scan', 'benign', 'floor', 'Benign'),
    ]

    for scan_type, sub_cat, sub_sub, cls in scan_configs:
        base_path = os.path.join(JAMMING_DIR, scan_type, sub_cat, sub_sub)
        if not os.path.exists(base_path):
            continue

        csv_files = []
        for root, dirs, files in os.walk(base_path):
            for f in sorted(files):
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))

        if not csv_files:
            continue

        print(f"  Processing: {scan_type}/{sub_cat}/{sub_sub} -> '{cls}' ({len(csv_files)} files)")

        group_size = 8
        spectral_buffer = []

        for csv_path in csv_files:
            try:
                data_rows = []
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        try:
                            vals = [float(v) for v in row if v.strip()]
                            if vals:
                                data_rows.append(vals)
                        except ValueError:
                            continue

                if not data_rows:
                    continue

                arr = np.array(data_rows)
                power_vec = arr[:, -1] if arr.shape[1] > 1 else arr[:, 0]
                spectral_buffer.append(power_vec)

                if len(spectral_buffer) >= group_size:
                    min_len = min(len(v) for v in spectral_buffer[:group_size])
                    spec_2d = np.array([v[:min_len] for v in spectral_buffer[:group_size]])

                    out_path = os.path.join(STAGING_DIR, cls, f"jam_{cls}_{sample_idx:06d}.png")
                    if spectrogram_to_image(spec_2d, out_path):
                        counts[cls] = counts.get(cls, 0) + 1
                        sample_idx += 1
                    spectral_buffer = spectral_buffer[group_size:]

            except Exception:
                continue

        if len(spectral_buffer) >= 3:
            min_len = min(len(v) for v in spectral_buffer)
            spec_2d = np.array([v[:min_len] for v in spectral_buffer])
            out_path = os.path.join(STAGING_DIR, cls, f"jam_{cls}_{sample_idx:06d}.png")
            if spectrogram_to_image(spec_2d, out_path):
                counts[cls] = counts.get(cls, 0) + 1
                sample_idx += 1

    print(f"[OK] Jamming processing complete. Counts: {counts}")
    return counts


# --- PHASE 4: BALANCE & AUGMENT ---------------------------------------------

def augment_class(cls_dir, target_count, cls_name):
    """Augment using physically meaningful RF transformations."""
    existing = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
    current = len(existing)

    if current == 0:
        print(f"  [WARN] No images in {cls_name} to augment!")
        return
    if current >= target_count:
        print(f"  [skip] {cls_name}: {current} samples (>= target {target_count})")
        return

    needed = target_count - current
    print(f"  [aug] {cls_name}: {current} -> need {needed} more")

    np.random.seed(hash(cls_name) % 2**31)
    aug_idx = 0
    while aug_idx < needed:
        src_path = existing[np.random.randint(0, current)]
        img = np.array(Image.open(src_path).convert('L'), dtype=np.float32)
        aug_type = np.random.randint(0, 4)

        if aug_type == 0:  # Time shift
            shift = np.random.randint(-img.shape[1] // 4, img.shape[1] // 4)
            img = np.roll(img, shift, axis=1)
        elif aug_type == 1:  # Frequency shift
            shift = np.random.randint(-img.shape[0] // 8, img.shape[0] // 8)
            img = np.roll(img, shift, axis=0)
        elif aug_type == 2:  # Additive noise
            noise = np.random.normal(0, np.random.uniform(5, 20), img.shape)
            img = np.clip(img + noise, 0, 255)
        elif aug_type == 3:  # Amplitude scaling
            scale = np.random.uniform(0.8, 1.2)
            img = np.clip(img * scale, 0, 255)

        out_path = os.path.join(cls_dir, f"aug_{cls_name}_{aug_idx:06d}.png")
        Image.fromarray(img.astype(np.uint8), mode='L').save(out_path)
        aug_idx += 1

    print(f"  [OK] {cls_name}: augmented to {current + needed} samples")


def balance_dataset():
    """Undersample majority, augment minority in staging directory."""
    print("\n" + "=" * 60)
    print("PHASE 4: Balancing and Augmenting Dataset")
    print("=" * 60)

    print("\nPre-balance counts:")
    for cls in CLASSES:
        count = len(glob.glob(os.path.join(STAGING_DIR, cls, "*.png")))
        print(f"  {cls:20s}: {count:6d}")

    # Undersample
    print("\nUndersampling majority classes...")
    for cls in CLASSES:
        cls_dir = os.path.join(STAGING_DIR, cls)
        images = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
        if len(images) > TARGET_PER_CLASS:
            np.random.seed(42)
            keep = set(np.random.choice(len(images), TARGET_PER_CLASS, replace=False))
            removed = 0
            for i, p in enumerate(images):
                if i not in keep:
                    os.remove(p)
                    removed += 1
            print(f"  {cls}: removed {removed} (kept {TARGET_PER_CLASS})")

    # Augment
    print("\nAugmenting minority classes...")
    for cls in CLASSES:
        augment_class(os.path.join(STAGING_DIR, cls), TARGET_PER_CLASS, cls)

    print("\nPost-balance counts:")
    for cls in CLASSES:
        count = len(glob.glob(os.path.join(STAGING_DIR, cls, "*.png")))
        print(f"  {cls:20s}: {count:6d}")


# --- PHASE 5: TRAIN/VAL/TEST SPLIT ------------------------------------------

def split_dataset():
    """
    Split the balanced staging dataset into train/val/test (70/15/15).
    Uses stratified random split to ensure each class is proportionally
    represented in every split.
    """
    print("\n" + "=" * 60)
    print("PHASE 5: Splitting into Train / Val / Test (70/15/15)")
    print("=" * 60)

    np.random.seed(42)  # Reproducible splits

    total_counts = {'train': {}, 'val': {}, 'test': {}}

    for cls in CLASSES:
        src_dir = os.path.join(STAGING_DIR, cls)
        images = sorted(glob.glob(os.path.join(src_dir, "*.png")))

        if not images:
            print(f"  [WARN] {cls}: no images to split!")
            continue

        # Shuffle
        indices = np.arange(len(images))
        np.random.shuffle(indices)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # Test gets the remainder to avoid rounding issues
        n_test = n - n_train - n_val

        splits = {
            'train': indices[:n_train],
            'val': indices[n_train:n_train + n_val],
            'test': indices[n_train + n_val:],
        }

        for split_name, split_indices in splits.items():
            dest_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            for idx in split_indices:
                src = images[idx]
                dst = os.path.join(dest_dir, os.path.basename(src))
                shutil.copy2(src, dst)
            total_counts[split_name][cls] = len(split_indices)

        print(f"  {cls:20s}: train={total_counts['train'][cls]}, "
              f"val={total_counts['val'][cls]}, test={total_counts['test'][cls]}")

    # Summary table
    print("\n--- FINAL DATASET SUMMARY ---")
    print(f"  {'Class':20s} {'Train':>8s} {'Val':>8s} {'Test':>8s} {'Total':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    grand_train = grand_val = grand_test = 0
    for cls in CLASSES:
        tr = total_counts['train'].get(cls, 0)
        va = total_counts['val'].get(cls, 0)
        te = total_counts['test'].get(cls, 0)
        grand_train += tr; grand_val += va; grand_test += te
        print(f"  {cls:20s} {tr:8d} {va:8d} {te:8d} {tr+va+te:8d}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':20s} {grand_train:8d} {grand_val:8d} {grand_test:8d} "
          f"{grand_train+grand_val+grand_test:8d}")

    return total_counts


# --- PHASE 6: VERIFICATION --------------------------------------------------

def verify_dataset():
    """Verify final dataset and generate visualization."""
    print("\n" + "=" * 60)
    print("PHASE 6: Verification")
    print("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Unified RF Intrusion Detection Dataset -- Sample per Class (Train Set)',
                 fontsize=14)

    all_good = True
    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(OUTPUT_DIR, 'train', cls)
        images = sorted(glob.glob(os.path.join(cls_dir, "*.png")))

        if not images:
            print(f"  [FAIL] {cls}: NO IMAGES in train!")
            all_good = False
            continue

        img = Image.open(images[0]).convert('L')
        arr = np.array(img)

        if arr.shape != (IMG_SIZE, IMG_SIZE):
            print(f"  [FAIL] {cls}: Wrong shape {arr.shape}")
            all_good = False
        else:
            total = sum(len(glob.glob(os.path.join(OUTPUT_DIR, s, cls, "*.png")))
                       for s in ['train', 'val', 'test'])
            print(f"  [OK] {cls:20s}: {total:5d} total, shape={arr.shape}, "
                  f"range=[{arr.min()}, {arr.max()}]")

        ax = axes[idx // 3, idx % 3]
        ax.imshow(arr, cmap='viridis', aspect='auto')
        ax.set_title(f'{cls}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency / Doppler')

    plt.tight_layout()
    verify_path = os.path.join(OUTPUT_DIR, "dataset_verification.png")
    plt.savefig(verify_path, dpi=150)
    plt.close()
    print(f"\n[OK] Verification plot saved: {verify_path}")

    if all_good:
        print("[OK] All checks passed!")
    else:
        print("[FAIL] Some checks failed -- review above.")


# --- CLEANUP -----------------------------------------------------------------

def cleanup_staging():
    """Remove temporary staging directory."""
    print(f"\n[...] Cleaning up staging directory...")
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print("[OK] Staging directory removed.")


# --- MAIN --------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Unified RF Intrusion Detection Dataset Builder")
    print("  Classes: Machine, Human, Wildlife, Broadband_Jam,")
    print("           Narrowband_Jam, Benign")
    print("  Resolution: 224x224 | Split: 70/15/15")
    print("=" * 60)

    ensure_dirs()

    # Phase 1-3: Generate spectrograms into staging
    process_radar()
    process_dronerf()
    process_jamming()

    # Phase 4: Balance
    balance_dataset()

    # Phase 5: Split
    split_dataset()

    # Phase 6: Verify
    verify_dataset()

    # Cleanup
    cleanup_staging()

    print("\n" + "=" * 60)
    print("  DATASET BUILD COMPLETE!")
    print(f"  Output: {OUTPUT_DIR}")
    print("  Structure: train/ val/ test/ each with 6 class folders")
    print("=" * 60)


if __name__ == "__main__":
    main()
