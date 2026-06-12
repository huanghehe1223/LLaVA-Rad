import os

# 要在 import huggingface_hub 之前设置
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"

from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "lmsys/vicuna-7b-v1.5"
target_dir = Path("/kaggle/working/LLaVA-Rad/models/vicuna-7b-v1.5")
target_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(target_dir),
    etag_timeout=30,   # 这行是元数据请求超时
)

print(f"done: {target_dir}")