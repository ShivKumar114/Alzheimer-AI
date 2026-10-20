#!/usr/bin/env python3
"""Train an image-only Alzheimer MRI classifier with ResNet18 transfer learning.

Expected dataset layout (ImageFolder format)::

    my_custom_data/Combined Dataset/
    ├── train/<class_name>/<image files>
    └── test/<class_name>/<image files>

The validation set is stratified from ``train``. The supplied ``test`` split is
used once, after selecting the best checkpoint by validation macro F1.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


IMAGE_SIZE = 224
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train and evaluate a ResNet18 classifier on MRI image folders."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "my_custom_data" / "Combined Dataset",
        help="Folder containing train/ and test/ class folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "checkpoints" / "mri_resnet18",
        help="Directory for the best checkpoint and metrics.json.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Defaults to 0 for reliable Windows multiprocessing.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Do not download/use ImageNet pretrained weights.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_train_val_indices(
    targets: list[int], val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split only the training-folder indices, preserving each class."""
    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(targets):
        by_class.setdefault(label, []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label].copy()
        if len(indices) < 2:
            raise ValueError(
                "Each training class needs at least 2 images to make a "
                f"train/validation split; class index {label} has {len(indices)}."
            )
        rng.shuffle(indices)
        val_count = max(1, round(len(indices) * val_fraction))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    data_dir = args.data_dir.expanduser().resolve()
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    for folder in (train_dir, test_dir):
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected dataset folder: {folder}")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Make separate ImageFolder instances so training augmentation never affects
    # validation. Their sorted sample lists and class mappings are identical.
    train_augmented = datasets.ImageFolder(train_dir, transform=train_transform)
    train_eval = datasets.ImageFolder(train_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)
    if train_augmented.class_to_idx != train_eval.class_to_idx:
        raise RuntimeError("Inconsistent class folders in the training dataset.")
    if test_dataset.class_to_idx != train_augmented.class_to_idx:
        raise ValueError(
            "Train and test folders must contain the same class subfolders. "
            f"Train: {train_augmented.class_to_idx}; test: {test_dataset.class_to_idx}"
        )
    if not train_augmented.samples or not test_dataset.samples:
        raise ValueError("The train and test folders must both contain images.")

    train_indices, val_indices = stratified_train_val_indices(
        train_augmented.targets, args.val_fraction, args.seed
    )
    common_loader_args: dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        Subset(train_augmented, train_indices), shuffle=True, **common_loader_args
    )
    val_loader = DataLoader(
        Subset(train_eval, val_indices), shuffle=False, **common_loader_args
    )
    # Keep the supplied test set separate from all training and model-selection
    # steps; it is iterated only once after restoring the best validation model.
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)
    return train_loader, val_loader, test_loader, train_augmented.classes


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all: list[int] = []
    predictions_all: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions_all.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets_all.extend(labels.detach().cpu().tolist())

    if not targets_all:
        raise ValueError("A data split is empty; check the dataset and split settings.")
    average_loss = total_loss / len(targets_all)
    accuracy = accuracy_score(targets_all, predictions_all)
    return average_loss, float(accuracy), targets_all, predictions_all


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    set_seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in this PyTorch environment.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Device: {device} (CUDA unavailable or disabled)")

    train_loader, val_loader, test_loader, class_names = make_loaders(args)
    print(
        f"Classes: {class_names}\n"
        f"Images: train={len(train_loader.dataset)}, "
        f"validation={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
    )

    weights = None if args.no_pretrained else models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pth"
    history: list[dict[str, float | int]] = []
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience = 5
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy, val_targets, val_predictions = run_epoch(
            model, val_loader, criterion, device
        )
        val_macro_f1 = float(classification_report(
            val_targets, val_predictions,
            labels=list(range(len(class_names))),
            output_dict=True, zero_division=0,
        )["macro avg"]["f1-score"])
        scheduler.step(val_macro_f1)
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_macro_f1": val_macro_f1,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.4f}, "
            f"macro-F1 {val_macro_f1:.4f}"
        )

        improved = (val_macro_f1 > best_val_f1) or (
            val_macro_f1 == best_val_f1 and val_loss < best_val_loss
        )
        if improved:
            best_val_f1 = val_macro_f1
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "class_to_idx": {name: index for index, name in enumerate(class_names)},
                "validation_macro_f1": best_val_f1,
                "validation_loss": best_val_loss,
                "image_size": IMAGE_SIZE,
                "normalization_mean": IMAGENET_MEAN,
                "normalization_std": IMAGENET_STD,
                "architecture": "torchvision.resnet18",
                "pretrained": not args.no_pretrained,
                "seed": args.seed,
            }, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {patience} epochs without validation improvement.")
                break

    # Restore the checkpoint chosen only by the train-derived validation split.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Final, one-time evaluation on the untouched supplied test split.
    test_loss, test_accuracy, test_targets, test_predictions = run_epoch(
        model, test_loader, criterion, device
    )
    report = classification_report(
        test_targets, test_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    metrics = {
        "architecture": "torchvision.resnet18",
        "device": str(device),
        "class_names": class_names,
        "split_sizes": {
            "train": len(train_loader.dataset),
            "validation": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_val_f1,
        "test": {
            "loss": test_loss,
            "accuracy": test_accuracy,
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1": report["macro avg"]["f1-score"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "confusion_matrix": confusion_matrix(
                test_targets, test_predictions,
                labels=list(range(len(class_names))),
            ).tolist(),
            "classification_report": report,
        },
        "history": history,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Metrics saved: {metrics_path}")
    print(
        f"Test: loss={test_loss:.4f}, accuracy={test_accuracy:.4f}, "
        f"macro-F1={metrics['test']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
