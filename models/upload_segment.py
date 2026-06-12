from pathlib import Path
from huggingface_hub import HfApi, create_repo

# 不推荐长期硬编码 token，别 commit 到 GitHub
HF_TOKEN = ""

REPO_ID = "huanghehe1223/CXR-CLASS"
LOCAL_FILE = Path("models/class_segment/best_checkpoint.pt")

# 上传到 Hugging Face dataset repo 里的路径
PATH_IN_REPO = "best_checkpoint.pt"

if not LOCAL_FILE.exists():
    raise FileNotFoundError(f"Local file not found: {LOCAL_FILE}")

api = HfApi(token=HF_TOKEN)

# 如果 dataset repo 已存在，exist_ok=True 不会报错
create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    exist_ok=True,
)

api.upload_file(
    path_or_fileobj=str(LOCAL_FILE),
    path_in_repo=PATH_IN_REPO,
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message="Upload class segmentation checkpoint",
)

print(f"Uploaded {LOCAL_FILE} to https://huggingface.co/datasets/{REPO_ID}/blob/main/{PATH_IN_REPO}")