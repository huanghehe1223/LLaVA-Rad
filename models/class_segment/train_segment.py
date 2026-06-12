import math
import random
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from PIL import Image
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


# ROOT = Path(__file__).resolve().parent
suffix = ""
ROOT = Path("/cxr-segment")
DINOV2_ROOT = ROOT / "model" / "rad-dino" / "dinov2"
WEIGHTS_PATH = ROOT / "model" / "rad-dino" / "backbone_compatible.safetensors"
DATA_ROOT = ROOT / "dataset"
WORK_DIR = ROOT / "output" / f"pneumothorax_upernet_dinov2{suffix}"

IMAGE_SIZE = 518
PATCH_SIZE = 14
TARGET_LAYERS = [2, 5, 8, 11]
IN_CHANNELS = [768, 768, 768, 768]
NUM_CLASSES = 2
SEG_OUT_CHANNELS = 1
CLASS_NAMES = ("background", "pneumothorax")
GPU_INDEX = 0
SEED = 42

TRAIN_BATCH_SIZE = 24
VAL_BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MIN_LR = 1e-6
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
DICE_WEIGHT = 1.0
FOCAL_WEIGHT = 1.0
MAX_GRAD_NORM = 1.0
SAVE_EVERY_EPOCH = False
PROCESS_SAVE_INTERVAL = 10

IMAGE_MEAN = (0.5307, 0.5307, 0.5307)
IMAGE_STD = (0.2583, 0.2583, 0.2583)

assert IMAGE_SIZE % PATCH_SIZE == 0


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
            self.lateral_convs.append(
                ConvModule(in_channel, out_channels, kernel_size=1, norm=False, act=False)
            )
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ChestXrayPneumothoraxDataset(Dataset):
    def __init__(self, split_dir: Path, image_size: int = IMAGE_SIZE) -> None:
        self.split_dir = split_dir
        self.image_size = image_size
        self.image_dir = split_dir / "images"
        self.mask_dir = split_dir / "masks"
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in {self.image_dir}")

        self.mask_paths = []
        missing_masks = []
        for image_path in self.image_paths:
            mask_path = self.mask_dir / image_path.name
            if not mask_path.exists():
                missing_masks.append(mask_path)
            self.mask_paths.append(mask_path)
        if missing_masks:
            raise FileNotFoundError(f"Missing mask files: {missing_masks[:5]}")

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
            ]
        )
        self.mask_transform = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image_tensor = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask_array = np.asarray(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy((mask_array > 0).astype(np.int64))

        return image_tensor, mask_tensor


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> torch.Tensor:
    logits = logits.squeeze(1)
    targets = targets.float()
    binary_cross_entropy = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    pt = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    modulating_factor = (1.0 - pt).pow(gamma)
    loss = alpha_factor * modulating_factor * binary_cross_entropy
    return loss.mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    probabilities = torch.sigmoid(logits.squeeze(1))
    targets = targets.float()
    probabilities = probabilities.reshape(probabilities.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    intersection = torch.sum(probabilities * targets, dim=1)
    denominator = torch.sum(probabilities.pow(2) + targets.pow(2), dim=1)
    dice_score = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice_score.mean()


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    probabilities = torch.sigmoid(logits.squeeze(1))
    foreground_pred = probabilities >= 0.5
    foreground_target = targets == 1

    tp = torch.logical_and(foreground_pred, foreground_target).sum().item()
    fp = torch.logical_and(foreground_pred, ~foreground_target).sum().item()
    fn = torch.logical_and(~foreground_pred, foreground_target).sum().item()
    tn = torch.logical_and(~foreground_pred, ~foreground_target).sum().item()

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)

    return {"dice": float(dice), "iou": float(iou), "acc": float(acc)}


def plot_process_metrics(
    epoch_history: list[Dict[str, Any]],
    output_dir: Path,
    tag: str | None = None,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{tag}" if tag else ""
    loss_plot_path = output_dir / f"loss_curve{suffix}.png"
    val_metrics_plot_path = output_dir / f"val_metrics_curve{suffix}.png"

    if epoch_history:
        epochs = [entry["epoch"] for entry in epoch_history]
        train_losses = [entry["train_loss"] for entry in epoch_history]
        val_losses = [entry["val"]["loss"] for entry in epoch_history]

        plt.figure(figsize=(9, 5))
        plt.plot(epochs, train_losses, marker="o", linewidth=2, label="train_loss")
        plt.plot(epochs, val_losses, marker="s", linewidth=2, label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train/Val Loss per Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_plot_path, dpi=200)
        plt.close()

        plt.figure(figsize=(9, 5))
        for metric_name, marker in (("dice", "o"), ("iou", "s"), ("acc", "^")):
            values = [entry["val"][metric_name] for entry in epoch_history]
            plt.plot(epochs, values, marker=marker, linewidth=2, label=f"val_{metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics per Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(val_metrics_plot_path, dpi=200)
        plt.close()

    return {
        "loss_curve": str(loss_plot_path),
        "val_metrics_curve": str(val_metrics_plot_path),
    }


def plot_test_metrics(
    test_metrics: Dict[str, float],
    output_dir: Path,
    tag: str | None = None,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    test_metrics_plot_path = output_dir / f"test_metrics_bar{suffix}.png"

    test_names = ["loss", "dice", "iou", "acc"]
    test_values = [float(test_metrics[name]) for name in test_names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(test_names, test_values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.title("Final Test Metrics")
    plt.ylabel("Value")
    upper = max(test_values + [1.0]) * 1.15
    plt.ylim(0.0, upper)
    for bar, value in zip(bars, test_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(test_metrics_plot_path, dpi=200)
    plt.close()

    return str(test_metrics_plot_path)


def build_process_payload(
    epoch_history: list[Dict[str, Any]],
    best_epoch: int,
    best_val_dice: float,
    process_plot_paths: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "config": {
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "target_layers": TARGET_LAYERS,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "val_batch_size": VAL_BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "min_lr": MIN_LR,
            "focal_alpha": FOCAL_ALPHA,
            "focal_gamma": FOCAL_GAMMA,
            "dice_weight": DICE_WEIGHT,
            "focal_weight": FOCAL_WEIGHT,
            "seed": SEED,
            "class_names": list(CLASS_NAMES),
        },
        "best": {
            "epoch": best_epoch,
            "val_dice": float(best_val_dice),
            "checkpoint": str(WORK_DIR / "best_checkpoint.pt"),
        },
        "history": epoch_history,
        "plots": process_plot_paths,
    }


def build_final_payload(
    epoch_history: list[Dict[str, Any]],
    best_epoch: int,
    best_val_dice: float,
    process_plot_paths: Dict[str, str],
    test_metrics: Dict[str, float],
    test_plot_path: str,
) -> Dict[str, Any]:
    payload = build_process_payload(epoch_history, best_epoch, best_val_dice, process_plot_paths)
    payload["test"] = {k: float(v) for k, v in test_metrics.items()}
    payload["plots"] = {
        **process_plot_paths,
        "test_metrics_bar": test_plot_path,
    }
    return payload


def save_metrics_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_checkpoint_trusted(path: Path, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class FrozenDinoV2UPerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.hub.load(str(DINOV2_ROOT), "dinov2_vitb14", source="local", pretrained=False)
        backbone_state_dict = load_file(str(WEIGHTS_PATH), device="cpu")
        self.backbone.load_state_dict(backbone_state_dict, strict=True)
        self.backbone.requires_grad_(False)
        self.backbone.eval()

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


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = ChestXrayPneumothoraxDataset(DATA_ROOT / "train")
    val_dataset = ChestXrayPneumothoraxDataset(DATA_ROOT / "val")
    test_dataset = ChestXrayPneumothoraxDataset(DATA_ROOT / "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = DICE_WEIGHT * dice_loss(logits, masks) + FOCAL_WEIGHT * sigmoid_focal_loss(logits, masks)

            probabilities = torch.sigmoid(logits.squeeze(1))
            foreground_pred = probabilities >= 0.5
            foreground_target = masks == 1

            total_tp += torch.logical_and(foreground_pred, foreground_target).sum().item()
            total_fp += torch.logical_and(foreground_pred, ~foreground_target).sum().item()
            total_fn += torch.logical_and(~foreground_pred, foreground_target).sum().item()
            total_tn += torch.logical_and(~foreground_pred, ~foreground_target).sum().item()

            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

    eps = 1e-7
    dice = (2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)
    iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    acc = (total_tp + total_tn + eps) / (total_tp + total_tn + total_fp + total_fn + eps)

    return {
        "loss": total_loss / max(total_samples, 1),
        "dice": float(dice),
        "iou": float(iou),
        "acc": float(acc),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_val_dice: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_dice": best_val_dice,
            "image_size": IMAGE_SIZE,
            "target_layers": TARGET_LAYERS,
        },
        path,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Please run this training script on a GPU runtime.")
    if torch.cuda.device_count() <= GPU_INDEX:
        raise RuntimeError(
            f"Requested GPU index {GPU_INDEX}, but only {torch.cuda.device_count()} CUDA device(s) are available."
        )

    set_seed(SEED)
    device = torch.device(f"cuda:{GPU_INDEX}")
    torch.cuda.set_device(device)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = build_dataloaders()
    model = FrozenDinoV2UPerNet().to(device)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_val_dice = -1.0
    best_epoch = -1
    epoch_history: list[Dict[str, Any]] = []
    best_checkpoint_path = WORK_DIR / "best_checkpoint.pt"
    last_checkpoint_path = WORK_DIR / "last_checkpoint.pt"
    final_metrics_json_path = WORK_DIR / "metrics_history.json"
    use_amp = True

    tqdm.write(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    tqdm.write(f"Train samples: {len(train_loader.dataset)}")
    tqdm.write(f"Val samples: {len(val_loader.dataset)}")
    tqdm.write(f"Test samples: {len(test_loader.dataset)}")
    tqdm.write(f"Image size: {IMAGE_SIZE} x {IMAGE_SIZE}")
    tqdm.write(f"Classes: {CLASS_NAMES}")
    tqdm.write(f"Trainable parameters: {sum(param.numel() for param in trainable_params):,}")

    epoch_progress = tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_progress:
        model.train()
        running_loss = 0.0
        running_samples = 0

        train_progress = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Train {epoch:03d}/{NUM_EPOCHS:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for step, (images, masks) in enumerate(train_progress, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss_dice = dice_loss(logits, masks)
                loss_focal = sigmoid_focal_loss(logits, masks)
                loss = DICE_WEIGHT * loss_dice + FOCAL_WEIGHT * loss_focal

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            running_samples += images.size(0)
            current_lr = optimizer.param_groups[0]["lr"]
            average_loss = running_loss / max(running_samples, 1)
            train_progress.set_postfix(loss=f"{average_loss:.4f}", lr=f"{current_lr:.2e}")

        train_progress.close()

        train_loss = running_loss / max(running_samples, 1)
        val_metrics = evaluate(model, val_loader, device)
        current_val_dice = val_metrics["dice"]
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "lr": float(current_lr),
                "val": {k: float(v) for k, v in val_metrics.items()},
            }
        )

        tqdm.write(
            f"Epoch {epoch:03d} done | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | val_acc={val_metrics['acc']:.4f}"
        )
        epoch_progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_metrics['loss']:.4f}",
            val_dice=f"{val_metrics['dice']:.4f}",
            lr=f"{current_lr:.2e}",
        )

        save_checkpoint(last_checkpoint_path, model, optimizer, scheduler, scaler, epoch, best_val_dice)
        if SAVE_EVERY_EPOCH:
            epoch_checkpoint_path = WORK_DIR / f"epoch_{epoch:03d}.pt"
            save_checkpoint(epoch_checkpoint_path, model, optimizer, scheduler, scaler, epoch, best_val_dice)

        if current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            best_epoch = epoch
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, scaler, epoch, best_val_dice)
            tqdm.write(f"New best checkpoint saved with val_dice={best_val_dice:.4f}")

        if epoch % PROCESS_SAVE_INTERVAL == 0 and epoch != NUM_EPOCHS:
            process_plot_paths = plot_process_metrics(epoch_history, WORK_DIR, tag=f"epoch_{epoch:03d}")
            process_json_path = WORK_DIR / f"metrics_process_epoch_{epoch:03d}.json"
            process_payload = build_process_payload(epoch_history, best_epoch, best_val_dice, process_plot_paths)
            save_metrics_json(process_json_path, process_payload)
            tqdm.write(
                f"Saved process snapshot at epoch {epoch:03d} | "
                f"json={process_json_path} | "
                f"plots={process_plot_paths['loss_curve']}, {process_plot_paths['val_metrics_curve']}"
            )

        scheduler.step()

    epoch_progress.close()

    tqdm.write("Training finished. Loading best checkpoint for final test evaluation.")
    best_state = load_checkpoint_trusted(best_checkpoint_path, device)
    model.load_state_dict(best_state["model"], strict=True)
    test_metrics = evaluate(model, test_loader, device)
    tqdm.write(
        f"Test metrics | loss={test_metrics['loss']:.4f} | dice={test_metrics['dice']:.4f} | "
        f"iou={test_metrics['iou']:.4f} | acc={test_metrics['acc']:.4f}"
    )

    final_process_plot_paths = plot_process_metrics(epoch_history, WORK_DIR)
    final_test_plot_path = plot_test_metrics(test_metrics, WORK_DIR)

    final_metrics_payload = build_final_payload(
        epoch_history,
        best_epoch,
        best_val_dice,
        final_process_plot_paths,
        test_metrics,
        final_test_plot_path,
    )
    save_metrics_json(final_metrics_json_path, final_metrics_payload)
    tqdm.write(f"Saved metrics JSON to: {final_metrics_json_path}")
    tqdm.write(
        "Saved plots to: "
        f"{final_process_plot_paths['loss_curve']}, {final_process_plot_paths['val_metrics_curve']}, {final_test_plot_path}"
    )

