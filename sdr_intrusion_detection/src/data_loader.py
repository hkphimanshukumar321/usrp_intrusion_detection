import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Standard IEEE Classes from our unified dataset
CLASS_NAMES = ['Machine', 'Human', 'Wildlife', 'Broadband_Jam', 'Narrowband_Jam', 'Benign']
NUM_CLASSES = len(CLASS_NAMES)

def get_dataloaders(
    dataset_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
):
    """
    Creates PyTorch DataLoaders for the Unified SDR Dataset.
    Loads pre-generated STFT Spectrogram images from train/val/test folders.
    """
    
    # We expand the 1-channel grayscale STFT to 3-channels to perfectly match 
    # the expected input dimensions of pre-trained ImageNet models (ResNet, DenseNet, etc.)
    
    # Training augmentations to prevent overfitting on spectrograms
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5), # Time-reversal is valid augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Block out freq/time bands
    ])

    # Validation/Test only resize and normalize
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}. Expected train/val/test folders.")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transform)

    # Ensure class to index mapping matches our intended order
    print(f"Dataset Class Mapping: {train_dataset.class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Loaded Dataset: Train={len(train_dataset)} | Val={len(val_dataset)} | Test={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Quick Test
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "unified_dataset"))
    if os.path.exists(base_dir):
        train_l, val_l, test_l = get_dataloaders(base_dir, batch_size=16)
        x, y = next(iter(train_l))
        print(f"Batch Input Shape: {x.shape} (Expected: [16, 3, 224, 224])")
        print(f"Batch Label Shape: {y.shape}")
