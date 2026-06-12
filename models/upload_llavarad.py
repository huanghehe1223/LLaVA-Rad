from huggingface_hub import HfApi

# 不推荐长期硬编码；用完记得别提交到 GitHub
HF_TOKEN = ""

repo_id = "huanghehe1223/CXR-CLASS"
local_folder = "models/llava-rad"

api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="llava-rad",
    commit_message="Upload llava-rad LoRA files",
)

print("Upload finished!")