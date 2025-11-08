from huggingface_hub import HfApi, create_repo, upload_folder, create_tag, HfHubHTTPError
from pathlib import Path

HF_NAMESPACE = "KavitShah1998"  # change later to org name
BASE_DIR = Path.home() / ".cache" / "huggingface" / "lerobot" / "argus_assignment"
DATASET_NAME_PREFIX = "nov_6_dataset_final"

api = HfApi()

for dataset_dir in BASE_DIR.glob(f"{DATASET_NAME_PREFIX}_*"):
    if not dataset_dir.is_dir():
        continue

    repo_name = dataset_dir.name
    repo_id = f"{HF_NAMESPACE}/{repo_name}"

    print("======================================================")
    print(f"📦 Dataset: {dataset_dir}")
    print(f"➡️  Repo:    {repo_id}")
    print("======================================================")

    # 1. Create repo if missing
    try:
        create_repo(repo_id, repo_type="dataset", private=True)
        print(f"✅ Created new repo: {repo_id}")
    except HfHubHTTPError as e:
        if "already exists" in str(e):
            print(f"ℹ️ Repo already exists: {repo_id}, continuing.")
        else:
            raise

    # 2. Upload folder (sync mode — deletes missing files)
    print("⬆️ Uploading (syncing files to Hugging Face)...")
    upload_folder(
        folder_path=str(dataset_dir),
        repo_id=repo_id,
        repo_type="dataset",
        delete=True,              # sync behavior
        ignore_patterns=["*.tmp", "*.DS_Store"],
    )
    print("✅ Upload complete.")

    # 3. Tag version (fails silently if exists)
    try:
        create_tag(repo_id, tag="v3.0", repo_type="dataset")
        print("🏷️  Created tag: v3.0")
    except HfHubHTTPError:
        print("ℹ️ Tag v3.0 already exists, skipping.")

    print(f"✅ Finished: {repo_id}\n")

print("🎉 All datasets uploaded successfully!")

