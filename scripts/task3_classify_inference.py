#!/usr/bin/env python3
"""
Task 1: Classification inference using DINOv2 backbone + linear head.
Loads a radiology image and outputs class probabilities.

Usage:
    python scripts/task3_classify_inference.py
    python scripts/task3_classify_inference.py <image_path>
"""

import sys
import os
import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

# --- Hardcoded paths ---
MODEL_REPO = "/kaggle/working/LLaVA-Rad/models/rad-dino/dinov2"
WEIGHTS_PATH = "/kaggle/working/LLaVA-Rad/models/rad-dino/backbone_compatible.safetensors"
HEAD_CKPT = "/kaggle/working/LLaVA-Rad/models/class_segment/best_linear_head.pt"
DEFAULT_IMAGE = "/kaggle/working/LLaVA-Rad/class_example.png"

# --- Constants (must match train_classify.py) ---
MODEL_NAME = "dinov2_vitb14"
IMAGE_SIZE = 448
IMAGE_MEAN = (0.5307, 0.5307, 0.5307)
IMAGE_STD = (0.2583, 0.2583, 0.2583)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """Load checkpoint with compatible settings."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def resolve_weights_path() -> str:
    """Resolve backbone weights path — local file or HF Hub download."""
    local_path = "/kaggle/working/LLaVA-Rad/models/rad-dino/backbone_compatible.safetensors"
    if os.path.exists(local_path):
        return local_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="microsoft/rad-dino", filename="backbone_compatible.safetensors")


def load_safetensors(path: str) -> dict:
    """Load safetensors weights."""
    from safetensors.torch import load_file
    return load_file(path, device="cpu")


def build_model(device: torch.device):
    """Build DINOv2 backbone and load pretrained weights."""
    # Load backbone architecture
    backbone = torch.hub.load(MODEL_REPO, MODEL_NAME, source="local", pretrained=False)
    # Load pretrained weights
    weights_path = resolve_weights_path()
    print(f"  Backbone weights: {weights_path}")
    state_dict = load_safetensors(weights_path)
    backbone.load_state_dict(state_dict, strict=True)
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def build_head(device: torch.device) -> tuple[nn.Module, list[str], dict[str, int]]:
    """Load the trained linear head and class metadata."""
    ckpt = load_checkpoint(HEAD_CKPT, device)
    class_names = ckpt["class_names"]
    class_to_idx = ckpt["class_to_idx"]
    feature_dim = ckpt["feature_dim"]

    head = nn.Linear(feature_dim, len(class_names)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()
    return head, class_names, class_to_idx


def build_transform() -> transforms.Compose:
    """Build eval image transform (matches train_classify.py eval_transform)."""
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def extract_cls_feature(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Extract CLS token feature from the DINOv2 backbone."""
    outputs = backbone.forward_features(images)
    if isinstance(outputs, dict):
        if "x_norm_clstoken" in outputs:
            feats = outputs["x_norm_clstoken"]
        elif "x_prenorm" in outputs:
            feats = outputs["x_prenorm"]
        else:
            keys = ", ".join(outputs.keys())
            raise KeyError(f"Unexpected forward_features keys: {keys}")
    else:
        feats = outputs
    if feats.ndim == 3:
        feats = feats[:, 0, :]
    return feats


def main():
    parser = argparse.ArgumentParser(description="Classification inference with DINOv2 + linear head")
    parser.add_argument(
        "image_path", type=str, nargs="?", default=DEFAULT_IMAGE,
        help="Path to input radiology image",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load image ---
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    print(f"Image: {args.image_path} ({image.size[0]}x{image.size[1]})")

    # --- 2. Load model ---
    print("Loading backbone...")
    backbone = build_model(device)
    print("Loading linear head...")
    head, class_names, class_to_idx = build_head(device)
    print(f"Classes ({len(class_names)}): {class_names}")

    # --- 3. Preprocess image ---
    transform = build_transform()
    image_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    print(f"Input tensor: {tuple(image_tensor.shape)}")

    # --- 4. Inference ---
    with torch.inference_mode():
        feats = extract_cls_feature(backbone, image_tensor)
        logits = head(feats)
        probs = torch.softmax(logits, dim=1)

    # --- 5. Output results ---
    print()
    print("=" * 60)
    print("  Classification Results")
    print("=" * 60)
    for i, name in enumerate(class_names):
        p = probs[0, i].item()
        bar = "█" * int(p * 40)
        print(f"  {name:32s}  {p:.4f}  ({p*100:5.1f}%)  {bar}")

    best_idx = probs[0].argmax().item()
    print("-" * 60)
    print(f"  Predicted: {class_names[best_idx]}  (confidence: {probs[0, best_idx].item():.4f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
