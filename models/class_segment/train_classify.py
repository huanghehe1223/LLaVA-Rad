#!/usr/bin/env python3
"""Train and evaluate a linear probe on top of a frozen RAD-DINO backbone."""

from __future__ import annotations

import json
import pickle
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_IMAGE_SIZE = 448
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 50

DEFAULT_DATASET_DIR = Path("/cxr-class/dataset")
DEFAULT_MODEL_REPO = Path("/cxr-class/model/rad-dino/dinov2")
DEFAULT_WEIGHTS_PATH = Path("/cxr-class/model/rad-dino/backbone_compatible.safetensors")
CFG_MODEL_NAME = "dinov2_vitb14"
CFG_OUTPUT_DIR = Path(f"/cxr-class/output/epoch_{DEFAULT_EPOCHS}")


CFG_NUM_WORKERS = 8

CFG_LR = 1e-4
CFG_WEIGHT_DECAY = 1e-3
CFG_ETA_MIN = 1e-6
CFG_LABEL_SMOOTHING = 0.0

CFG_GPU_INDEX = 0
CFG_SEED = 42
CFG_AMP = True
CFG_DISABLE_PROGRESS = False
CFG_EVAL_ONLY = False
CFG_CKPT_PATH = None



IMAGE_MEAN = (0.5307, 0.5307, 0.5307)
IMAGE_STD = (0.2583, 0.2583, 0.2583)

# ===== User Config (edit here directly) =====
CFG_DATASET_DIR = DEFAULT_DATASET_DIR
CFG_MODEL_REPO = DEFAULT_MODEL_REPO
CFG_WEIGHTS_PATH = DEFAULT_WEIGHTS_PATH


CFG_IMAGE_SIZE = DEFAULT_IMAGE_SIZE
CFG_BATCH_SIZE = DEFAULT_BATCH_SIZE
CFG_EPOCHS = DEFAULT_EPOCHS



@dataclass
class EvalResult:
    loss: float
    acc: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]]
    confusion_matrix: List[List[int]]


@dataclass
class TrainConfig:
    dataset_dir: Path
    model_repo: Path
    model_name: str
    weights_path: Path
    output_dir: Path
    image_size: int
    batch_size: int
    epochs: int
    num_workers: int
    lr: float
    weight_decay: float
    eta_min: float
    label_smoothing: float
    gpu_index: int
    seed: int
    amp: bool
    disable_progress: bool
    eval_only: bool
    ckpt_path: Path | None


def get_config() -> TrainConfig:
    ckpt_path = Path(CFG_CKPT_PATH) if CFG_CKPT_PATH is not None else None
    return TrainConfig(
        dataset_dir=Path(CFG_DATASET_DIR),
        model_repo=Path(CFG_MODEL_REPO),
        model_name=CFG_MODEL_NAME,
        weights_path=Path(CFG_WEIGHTS_PATH),
        output_dir=Path(CFG_OUTPUT_DIR),
        image_size=int(CFG_IMAGE_SIZE),
        batch_size=int(CFG_BATCH_SIZE),
        epochs=int(CFG_EPOCHS),
        num_workers=int(CFG_NUM_WORKERS),
        lr=float(CFG_LR),
        weight_decay=float(CFG_WEIGHT_DECAY),
        eta_min=float(CFG_ETA_MIN),
        label_smoothing=float(CFG_LABEL_SMOOTHING),
        gpu_index=int(CFG_GPU_INDEX),
        seed=int(CFG_SEED),
        amp=bool(CFG_AMP),
        disable_progress=bool(CFG_DISABLE_PROGRESS),
        eval_only=bool(CFG_EVAL_ONLY),
        ckpt_path=ckpt_path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_args_for_checkpoint(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for k, v in args_dict.items():
        if isinstance(v, Path):
            sanitized[k] = str(v)
        else:
            sanitized[k] = v
    return sanitized


def torch_load_compat(path: Path, map_location: torch.device, weights_only: bool) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint_compat(path: Path, map_location: torch.device) -> Dict[str, Any]:
    try:
        return torch_load_compat(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        print(
            "[WARN] Checkpoint contains non-tensor metadata incompatible with weights_only=True; "
            "falling back to weights_only=False for trusted local checkpoint."
        )
        return torch_load_compat(path, map_location=map_location, weights_only=False)


def validate_paths(args: TrainConfig) -> None:
    for split in ("train", "val", "test"):
        split_dir = args.dataset_dir / split
        if not split_dir.exists() or not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

    if not args.model_repo.exists() or not args.model_repo.is_dir():
        raise FileNotFoundError(f"Invalid model repo: {args.model_repo}")

    if not args.weights_path.exists() or not args.weights_path.is_file():
        raise FileNotFoundError(f"Invalid weights path: {args.weights_path}")


def ensure_output_dir(args: TrainConfig) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)


def plot_training_history(history: List[Dict[str, float]], output_dir: Path) -> Path | None:
    if plt is None:
        print("[WARN] matplotlib is not installed; skipping training history plots.")
        return None
    if not history:
        return None

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]
    val_macro_f1 = [row["val_macro_f1"] for row in history]
    learning_rate = [row["lr"] for row in history]
    epoch_sec = [row["epoch_sec"] for row in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0, 0].plot(epochs, val_loss, marker="o", label="val_loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(alpha=0.25)
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[0, 1].plot(epochs, val_acc, marker="o", label="val_acc")
    axes[0, 1].plot(epochs, val_macro_f1, marker="o", label="val_macro_f1")
    axes[0, 1].set_title("Accuracy / Macro F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, learning_rate, marker="o", color="tab:green")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("LR")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(epochs, epoch_sec, marker="o", color="tab:purple")
    axes[1, 1].set_title("Epoch Time")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].grid(alpha=0.25)

    fig.tight_layout()
    plot_path = output_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def plot_test_results(
    test_res: EvalResult,
    class_names: List[str],
    output_dir: Path,
    prefix: str,
) -> List[Path]:
    if plt is None:
        print("[WARN] matplotlib is not installed; skipping test-result plots.")
        return []

    saved_paths: List[Path] = []
    display_class_names: List[str] = []
    alias_notes: List[str] = []

    for name in class_names:
        if name == "No_Lung_Opacity_Not_Normal":
            display_class_names.append("NLONN")
            alias_notes.append("NLONN = No_Lung_Opacity_Not_Normal")
        else:
            display_class_names.append(name)

    alias_note_text = ""
    if alias_notes:
        alias_note_text = "; ".join(sorted(set(alias_notes)))

    overall_names = ["acc", "macro_precision", "macro_recall", "macro_f1"]
    overall_vals = [
        test_res.acc,
        test_res.macro_precision,
        test_res.macro_recall,
        test_res.macro_f1,
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(overall_names, overall_vals, color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"])
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Test Overall Metrics")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, overall_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    overall_path = output_dir / f"{prefix}_overall_metrics.png"
    fig.savefig(overall_path, dpi=160)
    plt.close(fig)
    saved_paths.append(overall_path)

    if class_names:
        precision = [test_res.per_class[name]["precision"] for name in class_names]
        recall = [test_res.per_class[name]["recall"] for name in class_names]
        f1 = [test_res.per_class[name]["f1"] for name in class_names]

        x = list(range(len(class_names)))
        width = 0.25

        fig, ax = plt.subplots(figsize=(max(10, len(class_names) * 1.2), 6))
        bars_precision = ax.bar([i - width for i in x], precision, width=width, label="precision")
        bars_recall = ax.bar(x, recall, width=width, label="recall")
        bars_f1 = ax.bar([i + width for i in x], f1, width=width, label="f1")

        for bar_group in (bars_precision, bars_recall, bars_f1):
            for bar in bar_group:
                value = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + 0.012,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(display_class_names, rotation=45, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.set_title("Test Per-Class Metrics")
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        if alias_note_text:
            fig.text(0.5, 0.01, f"Note: {alias_note_text}", ha="center", va="bottom", fontsize=9)
            fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        else:
            fig.tight_layout()
        per_class_path = output_dir / f"{prefix}_per_class_metrics.png"
        fig.savefig(per_class_path, dpi=160)
        plt.close(fig)
        saved_paths.append(per_class_path)

    confmat = test_res.confusion_matrix
    if confmat:
        n = len(confmat)
        max_value = max(max(row) for row in confmat) if n > 0 else 0
        confmat_xlabel = "Predicted"
        if alias_note_text:
            confmat_xlabel = f"Predicted\nNote: {alias_note_text}"

        fig, ax = plt.subplots(figsize=(max(6.5, n * 1.2), max(6.0, n * 1.1)), constrained_layout=True)
        image = ax.imshow(confmat, cmap="Blues", aspect="auto")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Test Confusion Matrix")
        ax.set_xlabel(confmat_xlabel)
        ax.set_ylabel("True")
        ax.set_xticks(list(range(len(class_names))))
        ax.set_yticks(list(range(len(class_names))))
        ax.set_xticklabels(display_class_names, rotation=45, ha="right")
        ax.set_yticklabels(display_class_names)

        threshold = max_value / 2.0 if max_value > 0 else 0.0
        for i in range(n):
            for j in range(len(confmat[i])):
                value = confmat[i][j]
                ax.text(
                    j,
                    i,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=8,
                )

        confmat_path = output_dir / f"{prefix}_confusion_matrix.png"
        fig.savefig(confmat_path, dpi=180)
        plt.close(fig)
        saved_paths.append(confmat_path)

        normalized_confmat: List[List[float]] = []
        for row in confmat:
            row_sum = float(sum(row))
            denom = row_sum if row_sum > 0.0 else 1.0
            normalized_confmat.append([float(value) / denom for value in row])

        fig, ax = plt.subplots(figsize=(max(6.5, n * 1.2), max(6.0, n * 1.1)), constrained_layout=True)
        image = ax.imshow(normalized_confmat, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Test Confusion Matrix (Normalized by True Class)")
        ax.set_xlabel(confmat_xlabel)
        ax.set_ylabel("True")
        ax.set_xticks(list(range(len(class_names))))
        ax.set_yticks(list(range(len(class_names))))
        ax.set_xticklabels(display_class_names, rotation=45, ha="right")
        ax.set_yticklabels(display_class_names)

        for i in range(n):
            for j in range(len(normalized_confmat[i])):
                value = normalized_confmat[i][j]
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value > 0.5 else "black",
                    fontsize=8,
                )

        confmat_norm_path = output_dir / f"{prefix}_confusion_matrix_normalized.png"
        fig.savefig(confmat_norm_path, dpi=180)
        plt.close(fig)
        saved_paths.append(confmat_norm_path)

    return saved_paths


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    # Gentle augmentations tailored for chest X-rays.
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomRotation(degrees=7, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.08, contrast=0.08)],
                p=0.3,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(args: TrainConfig) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], Dict[str, int]]:
    train_transform, eval_transform = build_transforms(args.image_size)

    train_ds = datasets.ImageFolder(args.dataset_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(args.dataset_dir / "val", transform=eval_transform)
    test_ds = datasets.ImageFolder(args.dataset_dir / "test", transform=eval_transform)

    class_names = train_ds.classes
    class_to_idx = train_ds.class_to_idx

    if val_ds.class_to_idx != class_to_idx or test_ds.class_to_idx != class_to_idx:
        raise RuntimeError("Class-to-index mapping is inconsistent across train/val/test")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_names, class_to_idx


def load_compatible_safetensors(weights_path: Path) -> Dict[str, torch.Tensor]:
    try:
        from rad_dino.utils import safetensors_to_state_dict

        return safetensors_to_state_dict(str(weights_path))
    except Exception:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Cannot load safetensors. Install rad-dino package or safetensors: pip install safetensors"
            ) from exc
        return load_file(str(weights_path), device="cpu")


def build_backbone(args: TrainConfig, device: torch.device) -> nn.Module:
    model = torch.hub.load(str(args.model_repo), args.model_name, source="local", pretrained=False)
    state_dict = load_compatible_safetensors(args.weights_path)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def extract_cls_feature(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
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
    if feats.ndim != 2:
        raise RuntimeError(f"Expected [B, C] features, got shape {tuple(feats.shape)}")
    return feats


def infer_feature_dim(backbone: nn.Module, device: torch.device, image_size: int) -> int:
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        feats = extract_cls_feature(backbone, dummy)
    return int(feats.shape[-1])


def compute_metrics_from_confmat(confmat: torch.Tensor, class_names: List[str]) -> Dict[str, Dict[str, float]]:
    eps = 1e-12
    tp = confmat.diag().float()
    fp = confmat.sum(dim=0).float() - tp
    fn = confmat.sum(dim=1).float() - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    per_class: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i].item()),
            "recall": float(recall[i].item()),
            "f1": float(f1[i].item()),
            "support": int(confmat[i].sum().item()),
        }
    return per_class


def evaluate(
    backbone: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
    use_amp: bool,
) -> EvalResult:
    head.eval()
    confmat = torch.zeros(len(class_names), len(class_names), dtype=torch.long)

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    autocast_enabled = use_amp and device.type == "cuda"

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                feats = extract_cls_feature(backbone, images)
                logits = head(feats)
                loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)

            bs = labels.size(0)
            total_loss += float(loss.item()) * bs
            total_samples += bs
            total_correct += int((preds == labels).sum().item())

            idx = labels * len(class_names) + preds
            confmat += torch.bincount(idx, minlength=len(class_names) ** 2).reshape(len(class_names), len(class_names)).cpu()

    per_class = compute_metrics_from_confmat(confmat, class_names)
    macro_precision = sum(v["precision"] for v in per_class.values()) / len(class_names)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(class_names)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(class_names)

    return EvalResult(
        loss=total_loss / max(total_samples, 1),
        acc=total_correct / max(total_samples, 1),
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=per_class,
        confusion_matrix=confmat.tolist(),
    )


def train_one_epoch(
    backbone: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epoch_idx: int,
    total_epochs: int,
    show_progress: bool,
) -> Tuple[float, float]:
    head.train()
    backbone.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    autocast_enabled = use_amp and device.type == "cuda"

    iterator = dataloader
    if show_progress:
        iterator = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Train {epoch_idx}/{total_epochs}",
            leave=False,
            dynamic_ncols=True,
        )

    for step, (images, labels) in enumerate(iterator, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, enabled=autocast_enabled):
                feats = extract_cls_feature(backbone, images)

        with torch.autocast(device_type=device.type, enabled=autocast_enabled):
            logits = head(feats)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = logits.argmax(dim=1)
        bs = labels.size(0)

        total_loss += float(loss.item()) * bs
        total_correct += int((preds == labels).sum().item())
        total_samples += bs

        if show_progress:
            avg_loss = total_loss / max(total_samples, 1)
            avg_acc = total_correct / max(total_samples, 1)
            iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}", step=f"{step}/{len(dataloader)}")

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def main() -> int:
    args = get_config()
    validate_paths(args)
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Please run on a GPU runtime.")
    if torch.cuda.device_count() <= args.gpu_index:
        raise RuntimeError(
            f"Requested gpu-index={args.gpu_index}, but only {torch.cuda.device_count()} CUDA device(s) exist"
        )

    device = torch.device(f"cuda:{args.gpu_index}")
    torch.cuda.set_device(device)

    ensure_output_dir(args)

    train_loader, val_loader, test_loader, class_names, class_to_idx = build_dataloaders(args)

    backbone = build_backbone(args, device)
    feature_dim = infer_feature_dim(backbone, device, args.image_size)

    head = nn.Linear(feature_dim, len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    if args.eval_only:
        eval_ckpt = args.ckpt_path if args.ckpt_path is not None else (args.output_dir / "best_linear_head.pt")
        if not eval_ckpt.exists() or not eval_ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found for eval-only mode: {eval_ckpt}")

        ckpt = load_checkpoint_compat(eval_ckpt, map_location=device)
        if "head_state_dict" not in ckpt:
            raise KeyError(f"Invalid checkpoint, missing 'head_state_dict': {eval_ckpt}")

        head.load_state_dict(ckpt["head_state_dict"])

        test_res = evaluate(
            backbone=backbone,
            head=head,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            use_amp=args.amp,
        )

        eval_summary = {
            "mode": "eval_only",
            "checkpoint": str(eval_ckpt),
            "test_loss": float(test_res.loss),
            "test_acc": float(test_res.acc),
            "test_macro_precision": float(test_res.macro_precision),
            "test_macro_recall": float(test_res.macro_recall),
            "test_macro_f1": float(test_res.macro_f1),
            "test_per_class": test_res.per_class,
            "test_confusion_matrix": test_res.confusion_matrix,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
        }

        summary_path = args.output_dir / "metrics_summary_eval_only.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2, ensure_ascii=False)

        print("\n=== Eval Only (Test) ===")
        print(f"Checkpoint: {eval_ckpt}")
        print(f"Test loss: {test_res.loss:.4f}")
        print(f"Test acc: {test_res.acc:.4f}")
        print(f"Test macro precision: {test_res.macro_precision:.4f}")
        print(f"Test macro recall: {test_res.macro_recall:.4f}")
        print(f"Test macro f1: {test_res.macro_f1:.4f}")
        print(f"Saved metrics: {summary_path}")

        eval_plot_paths = plot_test_results(
            test_res=test_res,
            class_names=class_names,
            output_dir=args.output_dir,
            prefix="eval_only_test",
        )
        for path in eval_plot_paths:
            print(f"Saved plot: {path}")

        for cls in class_names:
            stats = test_res.per_class[cls]
            print(
                f"  [{cls}] precision={stats['precision']:.4f} "
                f"recall={stats['recall']:.4f} f1={stats['f1']:.4f} "
                f"support={stats['support']}"
            )

        return 0

    history: List[Dict[str, float]] = []
    args_for_ckpt = sanitize_args_for_checkpoint(vars(args))
    best_val_macro_f1 = -1.0
    best_ckpt = args.output_dir / "best_linear_head.pt"
    last_ckpt = args.output_dir / "last_linear_head.pt"

    print("=== Setup ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Backbone: {args.model_name}")
    print(f"Feature dim: {feature_dim}")
    print(f"Classes: {class_names}")
    print(f"Class to idx: {class_to_idx}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Scheduler: CosineAnnealingLR(T_max={args.epochs}, eta_min={args.eta_min})")
    print(f"AMP enabled: {args.amp}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            backbone=backbone,
            head=head,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=args.amp,
            epoch_idx=epoch,
            total_epochs=args.epochs,
            show_progress=not args.disable_progress,
        )

        val_res = evaluate(
            backbone=backbone,
            head=head,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            class_names=class_names,
            use_amp=args.amp,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        epoch_sec = time.time() - t0

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_res.loss),
            "val_acc": float(val_res.acc),
            "val_macro_f1": float(val_res.macro_f1),
            "lr": float(lr_now),
            "epoch_sec": float(epoch_sec),
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_res.loss:.4f} val_acc={val_res.acc:.4f} "
            f"val_macro_f1={val_res.macro_f1:.4f} lr={lr_now:.6e} time={epoch_sec:.1f}s"
        )

        ckpt_payload = {
            "epoch": epoch,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_val_macro_f1,
            "class_names": class_names,
            "class_to_idx": class_to_idx,
            "feature_dim": feature_dim,
            "args": args_for_ckpt,
        }
        torch.save(ckpt_payload, last_ckpt)

        if val_res.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_res.macro_f1
            ckpt_payload["best_val_macro_f1"] = best_val_macro_f1
            torch.save(ckpt_payload, best_ckpt)
            print(f"  -> New best checkpoint saved: {best_ckpt}")

    if not best_ckpt.exists():
        raise RuntimeError("Best checkpoint was not saved")

    best_state = load_checkpoint_compat(best_ckpt, map_location=device)
    head.load_state_dict(best_state["head_state_dict"])

    test_res = evaluate(
        backbone=backbone,
        head=head,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        use_amp=args.amp,
    )

    summary = {
        "best_val_macro_f1": float(best_val_macro_f1),
        "test_loss": float(test_res.loss),
        "test_acc": float(test_res.acc),
        "test_macro_precision": float(test_res.macro_precision),
        "test_macro_recall": float(test_res.macro_recall),
        "test_macro_f1": float(test_res.macro_f1),
        "test_per_class": test_res.per_class,
        "test_confusion_matrix": test_res.confusion_matrix,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "history": history,
    }

    summary_path = args.output_dir / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    training_plot_path = plot_training_history(history, args.output_dir)
    test_plot_paths = plot_test_results(
        test_res=test_res,
        class_names=class_names,
        output_dir=args.output_dir,
        prefix="final_test",
    )

    print("\n=== Final Test ===")
    print(f"Best val macro f1: {best_val_macro_f1:.4f}")
    print(f"Test loss: {test_res.loss:.4f}")
    print(f"Test acc: {test_res.acc:.4f}")
    print(f"Test macro precision: {test_res.macro_precision:.4f}")
    print(f"Test macro recall: {test_res.macro_recall:.4f}")
    print(f"Test macro f1: {test_res.macro_f1:.4f}")
    print(f"Saved last checkpoint: {last_ckpt}")
    print(f"Saved best checkpoint: {best_ckpt}")
    print(f"Saved metrics: {summary_path}")
    if training_plot_path is not None:
        print(f"Saved plot: {training_plot_path}")
    for path in test_plot_paths:
        print(f"Saved plot: {path}")

    for cls in class_names:
        stats = test_res.per_class[cls]
        print(
            f"  [{cls}] precision={stats['precision']:.4f} "
            f"recall={stats['recall']:.4f} f1={stats['f1']:.4f} "
            f"support={stats['support']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
