from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "huanghehe1223/CXR-CLASS"
REPO_TYPE = "dataset"

REMOTE_FOLDER = "llava-rad"
LOCAL_FOLDER = Path("models/llava-rad")

api = HfApi()

LOCAL_FOLDER.mkdir(parents=True, exist_ok=True)

files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type=REPO_TYPE
)

target_files = [
    f for f in files
    if f.startswith(REMOTE_FOLDER + "/")
]

if not target_files:
    raise RuntimeError(f"No files found under remote folder: {REMOTE_FOLDER}/")

print(f"Found {len(target_files)} files. Downloading to {LOCAL_FOLDER} ...")

for remote_path in target_files:
    relative_path = remote_path[len(REMOTE_FOLDER) + 1:]
    local_path = LOCAL_FOLDER / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=remote_path,
        local_dir=LOCAL_FOLDER,
    )

    # hf_hub_download 会保留远程路径，所以会下载到 models/llava-rad/llava-rad/xxx
    # 这里把它移动成 models/llava-rad/xxx
    downloaded_path = Path(downloaded_path)
    if downloaded_path != local_path:
        local_path.write_bytes(downloaded_path.read_bytes())

    print(f"Downloaded: {remote_path} -> {local_path}")

# 清理可能多出来的 models/llava-rad/llava-rad/
extra_folder = LOCAL_FOLDER / REMOTE_FOLDER
if extra_folder.exists() and extra_folder.is_dir():
    import shutil
    shutil.rmtree(extra_folder)

print("Done!")