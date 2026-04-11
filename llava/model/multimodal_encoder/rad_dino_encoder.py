import os
from types import SimpleNamespace

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


RAD_DINO_TOWER_NAME = "rad-dino"
RAD_DINO_HF_REPO = "microsoft/rad-dino"
RAD_DINO_DEFAULT_MODEL_REPO = "dev/dinov2"
RAD_DINO_DEFAULT_CHECKPOINT = "dev/backbone_compatible.safetensors"
RAD_DINO_ENV_MODEL_REPO = "RAD_DINO_MODEL_REPO"
RAD_DINO_ENV_CHECKPOINT = "RAD_DINO_CHECKPOINT"
RAD_DINO_IMAGE_SIZE = 518
RAD_DINO_PATCH_SIZE = 14
RAD_DINO_IMAGE_MEAN = (0.5307, 0.5307, 0.5307)
RAD_DINO_IMAGE_STD = (0.2583, 0.2583, 0.2583)


class RadDINOImageProcessor:
    def __init__(self):
        self.image_mean = RAD_DINO_IMAGE_MEAN
        self.image_std = RAD_DINO_IMAGE_STD
        self.crop_size = {"height": RAD_DINO_IMAGE_SIZE, "width": RAD_DINO_IMAGE_SIZE}
        self.size = {"shortest_edge": RAD_DINO_IMAGE_SIZE}
        self._transform = transforms.Compose(
            [
                transforms.Resize(RAD_DINO_IMAGE_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop((RAD_DINO_IMAGE_SIZE, RAD_DINO_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )

    def preprocess(self, image, return_tensors="pt"):
        if return_tensors != "pt":
            raise NotImplementedError("RadDINOImageProcessor only supports return_tensors='pt'.")
        pixel_values = self._transform(image).unsqueeze(0)
        return {"pixel_values": pixel_values}

    def __call__(self, images, return_tensors="pt"):
        if return_tensors != "pt":
            raise NotImplementedError("RadDINOImageProcessor only supports return_tensors='pt'.")
        if isinstance(images, list):
            pixel_values = torch.stack([self._transform(image) for image in images], dim=0)
        else:
            pixel_values = self._transform(images).unsqueeze(0)
        return {"pixel_values": pixel_values}


class RadDinoVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, vision_tower_config=None, vision_tower_checkpoint=None):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, "mm_vision_select_layer", -1)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.model_repo = (
            vision_tower_config
            or getattr(args, "mm_vision_tower_config", None)
            or getattr(args, "vision_tower_config", None)
        )
        self.vision_tower_checkpoint = (
            vision_tower_checkpoint
            or getattr(args, "mm_vision_tower_checkpoint", None)
            or getattr(args, "vision_tower_checkpoint", None)
        )

        self.image_processor = RadDINOImageProcessor()

        self._hidden_size = 768
        self._num_patches = (RAD_DINO_IMAGE_SIZE // RAD_DINO_PATCH_SIZE) ** 2
        self._cfg_only = SimpleNamespace(
            hidden_size=self._hidden_size,
            image_size=RAD_DINO_IMAGE_SIZE,
            patch_size=RAD_DINO_PATCH_SIZE,
        )

        if not delay_load:
            self.load_model()

    def _resolve_checkpoint_path(self, checkpoint_path):
        env_checkpoint = os.getenv(RAD_DINO_ENV_CHECKPOINT)
        local_candidates = [checkpoint_path, env_checkpoint, RAD_DINO_DEFAULT_CHECKPOINT]
        for candidate in local_candidates:
            if candidate and os.path.exists(candidate):
                return candidate

        # Try HF Hub fallback by filename when local checkpoint is unavailable.
        from huggingface_hub import hf_hub_download

        filename_candidates = []
        if checkpoint_path:
            filename_candidates.append(os.path.basename(checkpoint_path))
        if env_checkpoint:
            filename_candidates.append(os.path.basename(env_checkpoint))
        filename_candidates.append(os.path.basename(RAD_DINO_DEFAULT_CHECKPOINT))

        seen = set()
        for filename in filename_candidates:
            if not filename or filename in seen:
                continue
            seen.add(filename)
            try:
                return hf_hub_download(repo_id=RAD_DINO_HF_REPO, filename=filename)
            except Exception:
                continue

        return None

    def _resolve_model_repo(self):
        env_model_repo = os.getenv(RAD_DINO_ENV_MODEL_REPO)
        local_candidates = [self.model_repo, env_model_repo, RAD_DINO_DEFAULT_MODEL_REPO]
        for candidate in local_candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _load_state_dict(self, checkpoint_path):
        if checkpoint_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(checkpoint_path)
        return torch.load(checkpoint_path, map_location="cpu")

    def load_model(self):
        resolved_model_repo = self._resolve_model_repo()
        if resolved_model_repo is None:
            raise FileNotFoundError(
                "Rad-DINO model repository not found. "
                f"Set --vision_tower_config or environment variable {RAD_DINO_ENV_MODEL_REPO}."
            )

        self.vision_tower = torch.hub.load(
            resolved_model_repo,
            "dinov2_vitb14",
            source="local",
            pretrained=False,
        )

        resolved_checkpoint = self._resolve_checkpoint_path(self.vision_tower_checkpoint)
        if resolved_checkpoint is None:
            raise FileNotFoundError(
                "No Rad-DINO checkpoint found locally or on HF Hub. "
                f"Set --vision_tower_checkpoint or environment variable {RAD_DINO_ENV_CHECKPOINT}."
            )

        state_dict = self._load_state_dict(resolved_checkpoint)
        self.vision_tower.load_state_dict(state_dict, strict=True)
        print("rad-dino loaded success!")
        self.vision_tower.requires_grad_(False)

        self._hidden_size = getattr(self.vision_tower, "embed_dim", getattr(self.vision_tower, "num_features", 768))
        patch_embed = getattr(self.vision_tower, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "num_patches"):
            self._num_patches = int(patch_embed.num_patches)

        self.is_loaded = True

    def _extract_features(self, images):
        features = self.vision_tower.forward_features(images)
        if "x_norm_patchtokens" not in features:
            raise KeyError("Rad-DINO forward_features output does not contain 'x_norm_patchtokens'.")

        patch_tokens = features["x_norm_patchtokens"]
        if self.select_feature == "patch":
            return patch_tokens
        if self.select_feature == "cls_patch":
            cls_token = features["x_norm_clstoken"].unsqueeze(1)
            return torch.cat([cls_token, patch_tokens], dim=1)
        raise ValueError(f"Unexpected select feature: {self.select_feature}")

    @torch.no_grad()
    def forward(self, images):
        if not self.is_loaded:
            self.load_model()

        if isinstance(images, list):
            image_features = []
            for image in images:
                image_dtype = image.dtype
                image_forward_out = self._extract_features(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0)
                )
                image_features.append(image_forward_out.to(image_dtype))
            return image_features

        image_dtype = images.dtype
        image_features = self._extract_features(images.to(device=self.device, dtype=self.dtype))
        return image_features.to(image_dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if not self.is_loaded:
            return torch.float32
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        if not self.is_loaded:
            return torch.device("cpu")
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self._cfg_only

    @property
    def hidden_size(self):
        return self._hidden_size

    @property
    def num_patches(self):
        return self._num_patches
