#!/usr/bin/env python3
"""
ChestAI 后端服务 — 封装分类、分割、报告生成三个模型为 REST API。

启动方式:
    python backend/main.py
    uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import math
import os
import hashlib
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from safetensors.torch import load_file as load_safetensors

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tflink import TFLinkClient

# ─────────────────────────────────────────────
# 项目根目录
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ─────────────────────────────────────────────
# 模型路径 (与脚本保持一致)
# ─────────────────────────────────────────────
CLASS_BACKBONE_REPO = str(MODELS_DIR / "rad-dino" / "dinov2")
CLASS_BACKBONE_WEIGHTS = str(MODELS_DIR / "rad-dino" / "backbone_compatible.safetensors")
CLASS_HEAD_CKPT = str(MODELS_DIR / "class_segment" / "best_linear_head.pt")

SEG_BACKBONE_REPO = str(MODELS_DIR / "rad-dino" / "dinov2")
SEG_BACKBONE_WEIGHTS = str(MODELS_DIR / "rad-dino" / "backbone_compatible.safetensors")
SEG_CKPT = str(MODELS_DIR / "class_segment" / "best_checkpoint.pt")

REPORT_MODEL_PATH = str(MODELS_DIR / "llava-rad")
REPORT_MODEL_BASE = str(MODELS_DIR / "vicuna-7b-v1.5")

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────
CLASS_IMAGE_SIZE = 448
CLASS_IMAGE_MEAN = (0.5307, 0.5307, 0.5307)
CLASS_IMAGE_STD = (0.2583, 0.2583, 0.2583)

SEG_IMAGE_SIZE = 518
SEG_PATCH_SIZE = 14
SEG_TARGET_LAYERS = [2, 5, 8, 11]
SEG_IN_CHANNELS = [768, 768, 768, 768]
SEG_OUT_CHANNELS = 1
SEG_IMAGE_MEAN = (0.5307, 0.5307, 0.5307)
SEG_IMAGE_STD = (0.2583, 0.2583, 0.2583)
SEG_THRESHOLD = 0.5

assert SEG_IMAGE_SIZE % SEG_PATCH_SIZE == 0

# ─────────────────────────────────────────────
# 全局模型缓存
# ─────────────────────────────────────────────
_device: torch.device | None = None
_classification_cache: dict | None = None  # {"backbone", "head", "class_names", "transform"}
_segmentation_cache: dict | None = None     # {"model", "transform"}
_report_cache: dict | None = None           # {"tokenizer", "model", "image_processor"}

# 简易内存存储 (auth / records)
_USERS: dict[str, dict] = {}               # username -> {id, username, password}
_TOKENS: dict[str, dict] = {}               # token -> user
_CLASS_RECORDS: list[dict] = []
_SEG_RECORDS: list[dict] = []
_REPORT_RECORDS: list[dict] = []


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时预加载全部模型。"""
    print("[Startup] Preloading all models...")
    _load_classification_model()
    _load_segmentation_model()
    _load_report_model()
    print("[Startup] All models loaded. Ready to serve.")
    yield
    # shutdown: nothing to clean up


app = FastAPI(title="ChestAI Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


# ═══════════════════════════════════════════
# 1. 分类模型
# ═══════════════════════════════════════════

def _load_classification_model() -> dict:
    """加载 DINOv2 backbone + 线性分类头，返回模型组件字典。"""
    global _classification_cache
    if _classification_cache is not None:
        return _classification_cache

    device = get_device()
    print(f"[Classification] Loading on {device} ...")

    # --- backbone ---
    backbone = torch.hub.load(CLASS_BACKBONE_REPO, "dinov2_vitb14", source="local", pretrained=False)
    sd = load_safetensors(CLASS_BACKBONE_WEIGHTS, device="cpu")
    backbone.load_state_dict(sd, strict=True)
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # --- linear head ---
    ckpt = torch.load(CLASS_HEAD_CKPT, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    feature_dim = ckpt["feature_dim"]
    head = nn.Linear(feature_dim, len(class_names)).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    # --- transform ---
    transform = transforms.Compose([
        transforms.Resize(CLASS_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop((CLASS_IMAGE_SIZE, CLASS_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLASS_IMAGE_MEAN, std=CLASS_IMAGE_STD),
    ])

    _classification_cache = {
        "backbone": backbone,
        "head": head,
        "class_names": class_names,
        "transform": transform,
    }
    print(f"[Classification] Loaded. Classes: {class_names}")
    return _classification_cache


def run_classification(image: Image.Image) -> dict:
    """对 PIL Image 运行分类推理，返回 {final_result, normal_prob, lung_opacity_prob, nlo_nn_prob}."""
    cache = _load_classification_model()
    backbone = cache["backbone"]
    head = cache["head"]
    class_names = cache["class_names"]
    transform = cache["transform"]
    device = get_device()

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = backbone.forward_features(image_tensor)
        if isinstance(outputs, dict):
            if "x_norm_clstoken" in outputs:
                feats = outputs["x_norm_clstoken"]
            elif "x_prenorm" in outputs:
                feats = outputs["x_prenorm"]
            else:
                raise KeyError(f"Unexpected forward_features keys: {list(outputs.keys())}")
        else:
            feats = outputs
        if feats.ndim == 3:
            feats = feats[:, 0, :]

        logits = head(feats)
        probs = torch.softmax(logits, dim=1)[0]

    # 按类别名提取概率
    prob_map = {}
    for i, name in enumerate(class_names):
        prob_map[name] = float(probs[i].item())

    best_idx = probs.argmax().item()
    final_result = class_names[best_idx]

    return {
        "final_result": final_result,
        "normal_prob": prob_map.get("Normal", 0.0),
        "lung_opacity_prob": prob_map.get("Lung_Opacity", 0.0),
        "nlo_nn_prob": prob_map.get("No_Lung_Opacity_Not_Normal", 0.0),
    }


# ═══════════════════════════════════════════
# 2. 分割模型
# ═══════════════════════════════════════════

def _build_norm_layer(num_channels: int) -> nn.GroupNorm:
    num_groups = math.gcd(32, num_channels)
    if num_groups == 0:
        num_groups = 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, norm: bool = True, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not norm)
        self.norm = _build_norm_layer(out_channels) if norm else None
        self.act = nn.ReLU(inplace=True) if act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int):
        super().__init__()
        self.pool_layers = nn.ModuleList()
        for scale in pool_scales:
            self.pool_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                ConvModule(in_channels, channels, kernel_size=1, norm=True, act=True),
            ))

    def forward(self, x: torch.Tensor):
        outs = []
        for pool_layer in self.pool_layers:
            pooled = pool_layer(x)
            pooled = F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
            outs.append(pooled)
        return tuple(outs)


class MultiLevelNeck(nn.Module):
    def __init__(self, in_channels, out_channels: int, scales=(0.5, 1.0, 2.0, 4.0)):
        super().__init__()
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.scales = list(scales)
        self.num_outs = len(self.scales)
        self.lateral_convs = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_ch in self.in_channels:
            self.lateral_convs.append(ConvModule(in_ch, out_channels, kernel_size=1, norm=False, act=False))
        for _ in range(self.num_outs):
            self.convs.append(ConvModule(out_channels, out_channels, kernel_size=3, padding=1, norm=False, act=False))

    def forward(self, inputs):
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
    def __init__(self, in_channels, channels: int, num_classes: int,
                 pool_scales=(1, 2, 3, 6), dropout_ratio: float = 0.1, align_corners: bool = False):
        super().__init__()
        self.in_channels = list(in_channels)
        self.channels = channels
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.psp_modules = PyramidPoolingModule(pool_scales, self.in_channels[-1], self.channels)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels, self.channels,
            kernel_size=3, padding=1, norm=True, act=True,
        )
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_ch in self.in_channels[:-1]:
            self.lateral_convs.append(ConvModule(in_ch, self.channels, kernel_size=1, norm=True, act=True))
            self.fpn_convs.append(ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm=True, act=True))
        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels, self.channels,
            kernel_size=3, padding=1, norm=True, act=True,
        )
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        self.conv_seg = nn.Conv2d(self.channels, self.num_classes, kernel_size=1)

    def cls_seg(self, feat: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            feat = self.dropout(feat)
        return self.conv_seg(feat)

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        return self.bottleneck(psp_outs)

    def _forward_feature(self, inputs):
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners,
            )
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        fpn_outs.append(laterals[-1])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners,
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        return self.fpn_bottleneck(fpn_outs)

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        return self.cls_seg(output)


class FrozenDinoV2UPerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(SEG_BACKBONE_REPO, "dinov2_vitb14", source="local", pretrained=False)
        self.neck = MultiLevelNeck(
            in_channels=SEG_IN_CHANNELS, out_channels=768, scales=[4, 2, 1, 0.5],
        )
        self.decode_head = UPerHead(
            in_channels=SEG_IN_CHANNELS, channels=512, num_classes=SEG_OUT_CHANNELS,
            pool_scales=(1, 2, 3, 6), dropout_ratio=0.1, align_corners=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone.get_intermediate_layers(
                images, n=SEG_TARGET_LAYERS, reshape=True, norm=True,
            )
        features = self.neck(features)
        logits = self.decode_head(features)
        return logits


def _load_segmentation_model() -> dict:
    """加载分割模型。"""
    global _segmentation_cache
    if _segmentation_cache is not None:
        return _segmentation_cache

    device = get_device()
    print(f"[Segmentation] Loading on {device} ...")

    model = FrozenDinoV2UPerNet().to(device)

    # 加载 backbone 权重
    backbone_sd = load_safetensors(SEG_BACKBONE_WEIGHTS, device="cpu")
    model.backbone.load_state_dict(backbone_sd, strict=True)

    # 加载完整 checkpoint
    ckpt = torch.load(SEG_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((SEG_IMAGE_SIZE, SEG_IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=SEG_IMAGE_MEAN, std=SEG_IMAGE_STD),
    ])

    _segmentation_cache = {"model": model, "transform": transform}
    print(f"[Segmentation] Loaded. epoch={ckpt['epoch']}, best_val_dice={ckpt['best_val_dice']:.4f}")
    return _segmentation_cache


def run_segmentation(image: Image.Image) -> str:
    """对 PIL Image 运行分割推理，保存 mask 并通过 TFLink 上传，返回下载链接。"""
    cache = _load_segmentation_model()
    model = cache["model"]
    transform = cache["transform"]
    device = get_device()

    original_size = image.size  # (W, H)
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image_tensor)
        logits = F.interpolate(
            logits, size=(original_size[1], original_size[0]),
            mode="bilinear", align_corners=False,
        )
        probs = torch.sigmoid(logits.squeeze(1))
        mask = (probs >= SEG_THRESHOLD).cpu().numpy().astype(np.uint8)[0]

    # 保存 mask 到临时文件
    mask_image = Image.fromarray(mask * 255, mode="L")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    try:
        mask_image.save(tmp_path)

        # 通过 TFLink 上传
        client = TFLinkClient()
        result = client.upload(tmp_path)
        download_link = result.download_link
        print(f"[Segmentation] Uploaded mask to: {download_link}")
        return download_link
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ═══════════════════════════════════════════
# 3. 报告生成模型 (LLaVA-Rad)
# ═══════════════════════════════════════════

def _load_report_model() -> dict:
    """加载 LLaVA-Rad 报告生成模型。"""
    global _report_cache
    if _report_cache is not None:
        return _report_cache

    print(f"[Report] Loading LLaVA-Rad model ...")

    # 延迟导入 llava 模块（避免与其它模型冲突）
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path

    disable_torch_init()

    model_name = get_model_name_from_path(REPORT_MODEL_PATH)
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path=REPORT_MODEL_PATH,
        model_base=REPORT_MODEL_BASE,
        model_name=model_name,
        device="cuda",
    )

    _report_cache = {
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
    }
    print(f"[Report] Loaded. model_name={model_name}")
    return _report_cache


def run_report(image: Image.Image) -> str:
    """对 PIL Image 运行报告生成，返回文本报告。"""
    cache = _load_report_model()
    tokenizer = cache["tokenizer"]
    model = cache["model"]
    image_processor = cache["image_processor"]

    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

    query = "Provide a description of the findings in the radiology image."

    # 构建 prompt
    if model.config.mm_use_im_start_end:
        from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = conv_templates["llava_v0"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 预处理图像
    image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()

    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # 停止条件
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # 推理
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    # 解码
    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    return outputs


# ═══════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════

# ── 健康检查 ──

@app.get("/api/health")
async def health():
    return {"status": "ok", "device": str(get_device())}


# ── 认证 (简易实现，前端需要) ──

@app.post("/api/auth/register")
async def register(payload: dict):
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    if not username or not password:
        raise HTTPException(400, "用户名和密码不能为空")
    if username in _USERS:
        raise HTTPException(400, "用户名已存在")

    user = {
        "id": int(hashlib.md5(username.encode()).hexdigest()[:12], 16) % 10**12,
        "username": username,
        "password": password,
    }
    _USERS[username] = user
    return {"message": "注册成功"}


@app.post("/api/auth/login")
async def login(payload: dict):
    username = (payload.get("username") or "").strip()
    password = (payload.get("password") or "").strip()
    user = _USERS.get(username)
    if not user or user["password"] != password:
        raise HTTPException(401, "用户名或密码错误")

    token = f"token-{username}-{int(time.time())}"
    _TOKENS[token] = user
    return {
        "token": token,
        "user": {"id": user["id"], "username": user["username"]},
    }


# ── 分类推理 ──

@app.post("/api/inference/classification")
async def api_classify(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图像文件")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "无法解析图像文件")

    try:
        result = run_classification(pil_image)
    except Exception as e:
        raise HTTPException(500, f"分类推理失败: {e}")

    # 保存记录
    record = {
        "classification_id": int(time.time() * 1000),
        "user_id": 0,
        "image_url": "",
        **result,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }
    _CLASS_RECORDS.insert(0, record)

    return result


# ── 分割推理 ──

@app.post("/api/inference/segmentation")
async def api_segment(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图像文件")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "无法解析图像文件")

    try:
        output_url = run_segmentation(pil_image)
    except Exception as e:
        raise HTTPException(500, f"分割推理失败: {e}")

    record = {
        "segmentation_id": int(time.time() * 1000),
        "user_id": 0,
        "image_url": "",
        "output_url": output_url,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }
    _SEG_RECORDS.insert(0, record)

    return {"output_url": output_url}


# ── 报告生成推理 ──

@app.post("/api/inference/report")
async def api_report(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "请上传图像文件")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "无法解析图像文件")

    try:
        output_report = run_report(pil_image)
    except Exception as e:
        raise HTTPException(500, f"报告生成失败: {e}")

    record = {
        "report_id": int(time.time() * 1000),
        "user_id": 0,
        "image_url": "",
        "output_report": output_report,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }
    _REPORT_RECORDS.insert(0, record)

    return {"output_report": output_report}


# ── 记录查询 ──

@app.get("/api/records/classification")
async def get_class_records():
    return _CLASS_RECORDS


@app.get("/api/records/segmentation")
async def get_seg_records():
    return _SEG_RECORDS


@app.get("/api/records/report")
async def get_report_records():
    return _REPORT_RECORDS


# ═══════════════════════════════════════════
# 启动入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9989)
