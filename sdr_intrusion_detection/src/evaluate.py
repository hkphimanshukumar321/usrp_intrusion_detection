import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.data_loader import get_dataloaders
from src.model import get_model
from src.config import (
    CLASS_NAMES, DEFAULT_DATA_DIR, DEFAULT_BATCH_SIZE, DEFAULT_NUM_WORKERS,
    CUSTOM_MODEL_NAME, CHECKPOINT_DIR, RESULTS_DIR,
)

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {args.model} on device: {device}")

    _, _, test_loader = get_dataloaders(
        dataset_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=DEFAULT_NUM_WORKERS
    )

    model = get_model(model_name=args.model).to(device)
    model_path = os.path.join(CHECKPOINT_DIR, f'best_{args.model}.pth')
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded checkpoint: {model_path}")
    else:
        print(f"[WARN] Checkpoint not found at {model_path}. Evaluating untrained model!")

    model.eval()
    all_preds = []
    all_targets = []

    print("\nRunning inference on Test Set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    print("\n" + "="*50)
    print(f"CLASSIFICATION REPORT: {args.model.upper()}")
    print("="*50)
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4))
    
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'Confusion Matrix: {args.model.upper()}')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, f'confusion_matrix_{args.model}.png')
    plt.savefig(cm_path)
    print(f"\nConfusion Matrix plot saved to: {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument('--model', type=str, default=CUSTOM_MODEL_NAME)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    evaluate_model(args)
