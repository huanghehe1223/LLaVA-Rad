import os
from .clip_encoder import CLIPVisionTower
from .open_clip_encoder import OpenCLIPVisionTower
from .rad_dino_encoder import RadDinoVisionTower, RAD_DINO_TOWER_NAME


RAD_DINO_TOWER_ALIASES = {RAD_DINO_TOWER_NAME, "rad_dino", "raddino"}


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    vision_tower_config = getattr(vision_tower_cfg, 'mm_vision_tower_config', getattr(vision_tower_cfg, 'vision_tower_config', None))
    vision_tower_checkpoint = getattr(vision_tower_cfg, 'mm_vision_tower_checkpoint', getattr(vision_tower_cfg, 'vision_tower_checkpoint', None))
    if vision_tower is None:
        vision_tower = RAD_DINO_TOWER_NAME

    vision_tower_lc = vision_tower.lower() if isinstance(vision_tower, str) else ""
    is_local_torch_hub_repo = (
        isinstance(vision_tower, str)
        and os.path.isdir(vision_tower)
        and os.path.exists(os.path.join(vision_tower, "hubconf.py"))
    )
    if vision_tower_lc in RAD_DINO_TOWER_ALIASES or is_local_torch_hub_repo:
        return RadDinoVisionTower(
            vision_tower,
            args=vision_tower_cfg,
            vision_tower_config=vision_tower_config,
            vision_tower_checkpoint=vision_tower_checkpoint,
            **kwargs,
        )

    is_absolute_path_exists = isinstance(vision_tower, str) and os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    elif vision_tower.startswith("hf-hub:") or vision_tower_config and vision_tower_checkpoint:
        return OpenCLIPVisionTower(
            vision_tower, args=vision_tower_cfg, vision_tower_config=vision_tower_config, vision_tower_checkpoint=vision_tower_checkpoint, **kwargs
        )

    raise ValueError(f'Unknown vision tower: {vision_tower}')
