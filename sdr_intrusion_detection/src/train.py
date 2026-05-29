import os
import json
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
from src.data_loader import get_dataloaders, CLASS_NAMES
from src.model import get_model


class FocalLoss(nn.Module):
    """Focal Loss with configurable gamma for hard-example mining."""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


def profile_model(model, device):
    """Calculates Params, Size, and Inference Latency."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = (total_params * 4) / (1024 ** 2)

    # Benchmark inference time
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(50):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "Total_Parameters": total_params,
        "Trainable_Parameters": trainable_params,
        "Model_Size_MB": round(model_size_mb, 2),
        "Inference_Time_ms": round(sum(times) / len(times), 2)
    }


def train_model(args):
    """Full training loop with W&B logging, profiling, and local JSON history."""

    wandb.init(
        project="SDR-Intrusion-Detection",
        name=f"{args.model}_lr{args.lr}_bs{args.batch_size}",
        config=vars(args),
        reinit=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        dataset_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4
    )

    model = get_model(model_name=args.model).to(device)

    # Profile
    print("\n--- MODEL PROFILING ---")
    profile_metrics = profile_model(model, device)
    for k, v in profile_metrics.items():
        print(f"  {k}: {v}")
    print("-----------------------\n")
    wandb.log(profile_metrics)

    # Optimization
    gamma = getattr(args, 'focal_gamma', 2.0)
    dropout = getattr(args, 'dropout', 0.3)
    weight_decay = getattr(args, 'weight_decay', 1e-4)

    criterion = FocalLoss(gamma=gamma)

    if getattr(args, 'optimizer', 'adamw') == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    best_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        with torch.no_grad():
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        scheduler.step()

        print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "learning_rate": scheduler.get_last_lr()[0]
        })

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            pth_path = f'checkpoints/best_{args.model}.pth'
            torch.save(model.state_dict(), pth_path)

            # ONNX export for edge deployment
            try:
                onnx_path = f'checkpoints/best_{args.model}.onnx'
                dummy = torch.randn(1, 3, 224, 224).to(device)
                torch.onnx.export(
                    model, dummy, onnx_path,
                    export_params=True, opset_version=14,
                    do_constant_folding=True,
                    input_names=['spectrogram'],
                    output_names=['class_logits'],
                    dynamic_axes={'spectrogram': {0: 'batch'}, 'class_logits': {0: 'batch'}}
                )
            except Exception:
                pass  # Some timm models don't export cleanly
            print(f"  [*] Best Model Saved (.pth + .onnx). Acc: {best_val_acc:.2f}%")

    # Save local JSON log
    log_data = {
        "model": args.model,
        "profile": profile_metrics,
        "best_val_acc": best_val_acc,
        "config": {
            "lr": args.lr, "batch_size": args.batch_size,
            "epochs": args.epochs,
            "focal_gamma": gamma,
            "optimizer": getattr(args, 'optimizer', 'adamw'),
            "weight_decay": weight_decay
        },
        "history": history
    }
    with open(f'results/logs/history_{args.model}.json', 'w') as f:
        json.dump(log_data, f, indent=4)

    wandb.finish()
    print(f"\nTraining Complete. Best Val Acc: {best_val_acc:.2f}%")
    return log_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=r'C:\Users\hkphi\OneDrive\Desktop\WORK\EndSemDSLabK\unified_dataset')
    parser.add_argument('--model', type=str, default='SDR_Custom_CoordASPP_Focal')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    train_model(args)
