from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "huanghehe1223/CXR-CLASS"
FILENAME = "best_checkpoint.pt"
LOCAL_DIR = Path("models/class_segment")

LOCAL_DIR.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    filename=FILENAME,
    local_dir=str(LOCAL_DIR),
)

print(f"Downloaded to: {path}")