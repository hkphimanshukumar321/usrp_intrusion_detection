import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import get_dataloaders, CLASS_NAMES
from src.model import get_model
import os

def evaluate_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {args.model} on device: {device}")

    _, _, test_loader = get_dataloaders(
        dataset_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )

    model = get_model(model_name=args.model).to(device)
    model_path = f'checkpoints/best_{args.model}.pth'
    
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
    
    os.makedirs('results', exist_ok=True)
    cm_path = f'results/confusion_matrix_{args.model}.png'
    plt.savefig(cm_path)
    print(f"\nConfusion Matrix plot saved to: {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'unified_dataset')))
    parser.add_argument('--model', type=str, default='SDR_Custom_CoordASPP_Focal')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    evaluate_model(args)
