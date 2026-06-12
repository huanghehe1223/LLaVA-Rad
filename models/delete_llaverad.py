from huggingface_hub import HfApi

# 不要提交到 GitHub；用完可以删掉这个脚本
HF_TOKEN = ""

repo_id = "huanghehe1223/CXR-CLASS"
remote_folder = "llava-rad"

api = HfApi(token=HF_TOKEN)

api.delete_folder(
    path_in_repo=remote_folder,
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Delete llava-rad folder",
)

print(f"Deleted remote folder: {remote_folder}")