#!/usr/bin/env python3
"""
Task 2: Segmentation inference using FrozenDinoV2UPerNet.
Loads a radiology image and outputs the predicted pneumothorax mask.

Usage:
    python scripts/task4_segment_inference.py
    python scripts/task4_segment_inference.py <image_path>
"""

import sys
import os
import math
import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from safetensors.torch import load_file

# --- Paths ---
MODEL_REPO = "/kaggle/working/LLaVA-Rad/models/rad-dino/dinov2"
CHECKPOINT_PATH = "/kaggle/working/LLaVA-Rad/models/class_segment/best_checkpoint.pt"
DEFAULT_IMAGE = "/kaggle/working/LLaVA-Rad/segment_example.png"
DEFAULT_OUTPUT = "/kaggle/working/LLaVA-Rad/segment_output.png"

# --- Constants (must match train_segment.py) ---
IMAGE_SIZE = 518
PATCH_SIZE = 14
TARGET_LAYERS = [2, 5, 8, 11]
IN_CHANNELS = [768, 768, 768, 768]
SEG_OUT_CHANNELS = 1
IMAGE_MEAN = (0.5307, 0.5307, 0.5307)
IMAGE_STD = (0.2583, 0.2583, 0.2583)
THRESHOLD = 0.5

assert IMAGE_SIZE % PATCH_SIZE == 0


# --- Model components (mirrors train_segment.py) ---


def build_norm_layer(num_channels: int) -> nn.GroupNorm:
    num_groups = math.gcd(32, num_channels)
    if num_groups == 0:
        num_groups = 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        norm: bool = True,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not norm)
        self.norm = build_norm_layer(out_channels) if norm else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.pool_layers = nn.ModuleList()
        for scale in pool_scales:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    ConvModule(in_channels, channels, kernel_size=1, norm=True, act=True),
                )
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs = []
        for pool_layer in self.pool_layers:
            pooled = pool_layer(x)
            pooled = F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
            outs.append(pooled)
        return tuple(outs)


class MultiLevelNeck(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int, ...] | list[int],
        out_channels: int,
        scales: Tuple[float, ...] | list[float] = (0.5, 1.0, 2.0, 4.0),
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.scales = list(scales)
        self.num_outs = len(self.scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in self.in_channels:
            self.lateral_convs.append(ConvModule(in_channel, out_channels, kernel_size=1, norm=False, act=False))
        for _ in range(self.num_outs):
            self.convs.append(ConvModule(out_channels, out_channels, kernel_size=3, padding=1, norm=False, act=False))

    def forward(self, inputs: Tuple[torch.Tensor, ...] | list[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        assert len(inputs) == len(self.in_channels)
        inputs = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        if len(inputs) == 1:
            inputs = [inputs[0] for _ in range(self.num_outs)]
        outs = []
        for i in range(self.num_outs):
            resized = F.interpolate(inputs[i], scale_factor=self.scales[i], mode="bilinear", align_corners=False)
            outs.append(self.convs[i](resized))
        return tuple(outs)


class UPerHead(nn.Module):
    def __init__(
        self,
        in_channels: Tuple[int, ...] | list[int],
        channels: int,
        num_classes: int,
        pool_scales: Tuple[int, ...] = (1, 2, 3, 6),
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.psp_modules = PyramidPoolingModule(pool_scales, self.in_channels[-1], self.channels)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm=True,
            act=True,
        )
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channel in self.in_channels[:-1]:
            self.lateral_convs.append(ConvModule(in_channel, self.channels, kernel_size=1, norm=True, act=True))
            self.fpn_convs.append(ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm=True, act=True))
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
            norm=True,
            act=True,
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def cls_seg(self, feat: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)

    def psp_forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def _forward_feature(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        return self.fpn_bottleneck(fpn_outs)

    def forward(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        output = self._forward_feature(inputs)
        return self.cls_seg(output)


class FrozenDinoV2UPerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            str(MODEL_REPO), "dinov2_vitb14", source="local", pretrained=False
        )
        self.neck = MultiLevelNeck(
            in_channels=IN_CHANNELS,
            out_channels=768,
            scales=[4, 2, 1, 0.5],
        )
        self.decode_head = UPerHead(
            in_channels=IN_CHANNELS,
            channels=512,
            num_classes=SEG_OUT_CHANNELS,
            pool_scales=(1, 2, 3, 6),
            dropout_ratio=0.1,
            align_corners=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                images,
                n=TARGET_LAYERS,
                reshape=True,
                norm=True,
            )
        features = self.neck(features)
        logits = self.decode_head(features)
        return logits


# --- Utilities ---


def resolve_backbone_weights() -> str:
    """Resolve backbone weights path."""
    local_path = "/kaggle/working/LLaVA-Rad/models/rad-dino/backbone_compatible.safetensors"
    if os.path.exists(local_path):
        return local_path
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id="microsoft/rad-dino", filename="backbone_compatible.safetensors")


def load_model(device: torch.device) -> FrozenDinoV2UPerNet:
    """Build model, load pretrained backbone, then load checkpoint."""
    print("Building model architecture...")
    model = FrozenDinoV2UPerNet().to(device)

    # Load backbone weights
    weights_path = resolve_backbone_weights()
    print(f"  Backbone weights: {weights_path}")
    backbone_sd = load_file(weights_path, device="cpu")
    model.backbone.load_state_dict(backbone_sd, strict=True)

    # Load full checkpoint (neck + decode_head + backbone from training)
    print(f"  Checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)

    model.eval()
    print(f"  Loaded from epoch {ckpt['epoch']}, best_val_dice={ckpt['best_val_dice']:.4f}")
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


def main():
    parser = argparse.ArgumentParser(description="Segmentation inference with FrozenDinoV2UPerNet")
    parser.add_argument(
        "image_path", type=str, nargs="?", default=DEFAULT_IMAGE,
        help="Path to input radiology image",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=DEFAULT_OUTPUT,
        help="Path to save the predicted mask",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Load image ---
    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    original_image = Image.open(args.image_path).convert("RGB")
    print(f"Image: {args.image_path} ({original_image.size[0]}x{original_image.size[1]})")

    # --- 2. Load model ---
    print("Loading model...")
    model = load_model(device)

    # --- 3. Preprocess ---
    transform = build_transform()
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    print(f"Input tensor: {tuple(image_tensor.shape)}")

    # --- 4. Inference ---
    print("Running inference...")
    with torch.inference_mode():
        logits = model(image_tensor)
        # Resize logits back to original image size
        logits = F.interpolate(
            logits,
            size=(original_image.size[1], original_image.size[0]),
            mode="bilinear",
            align_corners=False,
        )
        probs = torch.sigmoid(logits.squeeze(1))  # [1, H, W]
        mask = (probs >= THRESHOLD).cpu().numpy().astype(np.uint8)  # [1, H, W]

    # --- 5. Save mask ---
    mask_2d = mask[0] * 255  # binary mask: 0 or 255
    mask_image = Image.fromarray(mask_2d, mode="L")
    mask_image.save(args.output)
    print(f"Mask saved to: {args.output}")

    # --- 6. Summary ---
    foreground_ratio = mask_2d.mean() / 255.0 * 100
    print()
    print("=" * 60)
    print("  Segmentation Results")
    print("=" * 60)
    print(f"  Output shape:  {original_image.size[0]}x{original_image.size[1]}")
    print(f"  Foreground pixels (pneumothorax): {foreground_ratio:.2f}%")
    print(f"  Background pixels: {100 - foreground_ratio:.2f}%")
    print(f"  Mask saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
