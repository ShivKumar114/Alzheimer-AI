#!/usr/bin/env python3
"""Predict an Alzheimer MRI image class using the trained ResNet18 checkpoint.

Run from the repository root, for example:
    python predict_mri.py "my_custom_data/Combined Dataset/test/No Impairment/example.jpg"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one MRI image with the trained model.")
    parser.add_argument("image", type=Path, help="Path to a JPG, PNG, or other PIL-readable image.")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("checkpoints/mri_resnet18/best_model.pth"),
        help="Path to best_model.pth (default: checkpoints/mri_resnet18/best_model.pth).",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    with Image.open(image_path) as source:
        image = preprocess(source.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(image), dim=1)[0]
    confidence, class_index = probabilities.max(dim=0)
    print(f"Predicted class: {class_names[class_index.item()]}")
    print(f"Model confidence: {confidence.item() * 100:.2f}%")
    print("Class scores:")
    for name, score in sorted(
        zip(class_names, probabilities.tolist()), key=lambda item: item[1], reverse=True
    ):
        print(f"  {name}: {score * 100:.2f}%")


if __name__ == "__main__":
    main()
