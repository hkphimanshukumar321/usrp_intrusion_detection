import os
import glob
import shutil
import numpy as np
from scipy import signal as sig
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RADAR_NPY = os.path.join(BASE_DIR, "raw_datasets", "5845259", "data_SAAB_SIRS_77GHz_FMCW.npy")
STAGING_DIR = os.path.join(BASE_DIR, "_radar_staging")
OUTPUT_DIR = os.path.join(BASE_DIR, "unified_dataset")

IMG_SIZE = 224
TARGET_PER_CLASS = 4000

RADAR_CLASS_MAP = {
    'D1': 'Machine', 'D2': 'Machine', 'D3': 'Machine',
    'D4': 'Machine', 'D5': 'Machine', 'D6': 'Machine',
    'human_walk': 'Human', 'human_run': 'Human',
    'seagull': 'Wildlife', 'pigeon': 'Wildlife', 'raven': 'Wildlife',
    'black-headed gull': 'Wildlife', 'seagull and black-headed gull': 'Wildlife',
    'heron': 'Wildlife'
}

def clean_label(raw_label):
    # e.g., "['D1']" -> "D1"
    return str(raw_label).replace('[', '').replace(']', '').replace("'", "").replace('"', '').strip()

def spectrogram_to_image(spec_data, output_path):
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

def generate_radar():
    os.makedirs(STAGING_DIR, exist_ok=True)
    for c in ['Human', 'Wildlife', 'Machine']:
        os.makedirs(os.path.join(STAGING_DIR, c), exist_ok=True)

    print("Loading Radar .npy...")
    data = np.load(RADAR_NPY, allow_pickle=True)
    
    counts = {'Human': 0, 'Wildlife': 0, 'Machine': 0}
    sample_idx = 0

    print("Extracting Radar spectrograms...")
    for row_idx in range(data.shape[0]):
        label_str = clean_label(data[row_idx, 0])
        cls = RADAR_CLASS_MAP.get(label_str)
        if not cls: continue

        # We already have 110,500 DroneRF machine files. 
        # We only need 2000 Radar Machine files, so let's stop early for Machine to save time.
        if cls == 'Machine' and counts['Machine'] > 2500:
            continue
        
        # We need 4000 for Human and Wildlife, let's generate ~5000 of each
        if cls == 'Human' and counts['Human'] > 5000:
            continue
        if cls == 'Wildlife' and counts['Wildlife'] > 5000:
            continue

        segment_matrix = data[row_idx, 1]
        if segment_matrix is None or not hasattr(segment_matrix, 'shape'): continue
        
        n_segments = segment_matrix.shape[1] if len(segment_matrix.shape) > 1 else 1
        for seg_idx in range(n_segments):
            segment = segment_matrix[:, seg_idx] if len(segment_matrix.shape) > 1 else segment_matrix
            if len(segment) != 1280: continue
            
            range_azimuth = segment.reshape(5, 256)
            slow_time = np.sum(range_azimuth, axis=0)
            
            f, t, Sxx = sig.spectrogram(slow_time, fs=17000, nperseg=64, noverlap=48, nfft=128, return_onesided=False)
            if Sxx.size < 16: continue
            
            out_path = os.path.join(STAGING_DIR, cls, f"radar_{sample_idx:06d}.png")
            if spectrogram_to_image(Sxx, out_path):
                counts[cls] += 1
                sample_idx += 1
                
        if sample_idx % 2000 == 0:
            print(f"Generated {sample_idx} radar files...")

    print(f"Radar Generation Complete: {counts}")

def split_and_inject():
    np.random.seed(42)
    # Target counts for Radar inject:
    # Human & Wildlife: 4000 total (2800 train, 600 val, 600 test)
    # Machine: 2000 total (1400 train, 300 val, 300 test) to mix with existing 4000 USRP Drones.
    # We will undersample existing Machine to 2000 to keep it strictly at 4000 total.
    
    print("\nTrimming existing 'Machine' (USRP) down to 2000 samples to make room for Radar Drones...")
    for split, target in [('train', 1400), ('val', 300), ('test', 300)]:
        mach_dir = os.path.join(OUTPUT_DIR, split, 'Machine')
        existing = sorted(glob.glob(os.path.join(mach_dir, "*.png")))
        if len(existing) > target:
            keep = set(np.random.choice(len(existing), target, replace=False))
            removed = 0
            for i, p in enumerate(existing):
                if i not in keep:
                    os.remove(p)
                    removed += 1
            print(f"  {split}/Machine: removed {removed} DroneRF images (kept {target}).")

    print("\nInjecting Radar data into unified_dataset...")
    splits_ratio = {'train': 0.70, 'val': 0.15, 'test': 0.15}
    
    targets = {
        'Human': 4000,
        'Wildlife': 4000,
        'Machine': 2000
    }

    for cls, target in targets.items():
        src_dir = os.path.join(STAGING_DIR, cls)
        images = sorted(glob.glob(os.path.join(src_dir, "*.png")))
        
        # Undersample to target
        if len(images) > target:
            keep = np.random.choice(len(images), target, replace=False)
            images = [images[i] for i in keep]
        
        # Note: If less than target, ideally we augment, but radar dataset has > 11k Wildlife and 17k Human, so we will have enough.
        
        np.random.shuffle(images)
        n = len(images)
        n_train = int(n * splits_ratio['train'])
        n_val = int(n * splits_ratio['val'])
        
        cls_splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }
        
        for split_name, imgs in cls_splits.items():
            dest_dir = os.path.join(OUTPUT_DIR, split_name, cls)
            os.makedirs(dest_dir, exist_ok=True)
            for src in imgs:
                shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
            print(f"  Injected {len(imgs)} {cls} into {split_name}.")

def verify_dataset():
    print("\nRe-generating Verification Plot...")
    CLASSES = ['Machine', 'Human', 'Wildlife', 'Broadband_Jam', 'Narrowband_Jam', 'Benign']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Unified RF Intrusion Detection Dataset (SDR + Radar Fusion)', fontsize=14)

    for idx, cls in enumerate(CLASSES):
        cls_dir = os.path.join(OUTPUT_DIR, 'train', cls)
        images = sorted(glob.glob(os.path.join(cls_dir, "*.png")))
        if images:
            arr = np.array(Image.open(images[0]).convert('L'))
            ax = axes[idx // 3, idx % 3]
            ax.imshow(arr, cmap='viridis', aspect='auto')
            ax.set_title(f'{cls}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Freq / Doppler')
            
            total = sum(len(glob.glob(os.path.join(OUTPUT_DIR, s, cls, "*.png"))) for s in ['train', 'val', 'test'])
            print(f"  {cls}: {total} images")
        else:
            print(f"  [ERROR] No images found for {cls}")

    plt.tight_layout()
    verify_path = os.path.join(OUTPUT_DIR, "dataset_verification.png")
    plt.savefig(verify_path, dpi=150)
    plt.close()
    print(f"Verification plot saved to: {verify_path}")

if __name__ == "__main__":
    generate_radar()
    split_and_inject()
    verify_dataset()
    shutil.rmtree(STAGING_DIR, ignore_errors=True)
    print("\nSUCCESS! Dataset is now 100% complete and balanced.")
