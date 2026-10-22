"""Explicit, guarded upload of the local dataset package to Hugging Face Hub."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .dataset_export import validate_huggingface_package


def push_to_hub(hf_root: str | Path, repo_id: str, confirm_license_review: bool = False, create_repo: bool = False, private: bool = False) -> dict[str, Any]:
    if not confirm_license_review:
        raise PermissionError("Refusing upload: pass the explicit --confirm-license-review flag only after reviewing upstream NCM checkpoint, NCM code, XSum/source-data, and generated-output terms.")
    if not repo_id or "/" not in repo_id or repo_id.startswith("<"):
        raise ValueError("repo_id must be a real account/name value such as your-account/ncm-hierarchy-factuality-eval")
    root = Path(hf_root).resolve()
    check = validate_huggingface_package(root)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub before publishing") from exc
    api = HfApi()  # use the user's prior `hf auth login` credentials; never pass a token in source code
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Publish locally validated NCM hierarchy factuality dataset {check['manifest'].get('dataset_version', '')}".strip(),
        ignore_patterns=["*.sqlite3", "*.db", "*.log", "private/**", "**/.DS_Store", ".git/**", "**/.git/**", "__pycache__/**", "**/__pycache__/**"],
    )
    return {"repo_id": repo_id, "repo_type": "dataset", "uploaded": True, "dataset_version": check["manifest"].get("dataset_version")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", required=True, help="path to local NCM Research/huggingface directory")
    ap.add_argument("--repo-id", required=True, help="Hugging Face account/repository-name")
    ap.add_argument("--confirm-license-review", action="store_true", help="explicit acknowledgement that dataset/model/source/output terms were reviewed")
    ap.add_argument("--create-repo", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    result = push_to_hub(args.folder, args.repo_id, args.confirm_license_review, args.create_repo, args.private)
    print(f"Uploaded {result['repo_id']} (dataset version {result['dataset_version']}).")

if __name__ == "__main__":
    main()
