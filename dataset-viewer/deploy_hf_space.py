"""Prepare and optionally publish the private Qwen annotation viewer Space."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
ANNOTATIONS = PROJECT_ROOT / "notebook/dataset/taxonomy/taxonomy_work/annotations_v3_4_full_color.jsonl"
SPACE_NAME = "iris-pixelart-qwen38-viewer"


def load_token() -> str:
    for env_file in (
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "training/.env",
        Path.home() / "iris-sprite-lora/training/.env",
    ):
        if not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() in {"HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"}:
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN absent; il doit pouvoir lire le dataset privé et écrire dans un Space.")
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy", action="store_true", help="Publie le Space privé après préparation du paquet.")
    parser.add_argument("--repo-id", help="ID HF facultatif; par défaut <compte>/iris-pixelart-qwen38-viewer.")
    args = parser.parse_args()

    if not ANNOTATIONS.is_file():
        raise FileNotFoundError(f"JSONL Qwen3.8 introuvable : {ANNOTATIONS}")
    line_count = sum(1 for line in ANNOTATIONS.open("rb") if line.endswith(b"\n"))
    print(f"Annotations à inclure dans le Space privé : {line_count:,} lignes · {ANNOTATIONS.stat().st_size / 1024**2:.1f} Mo")
    if not args.deploy:
        print("Aucun fichier publié. Pour publier après revue : python deploy_hf_space.py --deploy")
        return

    token = load_token()
    api = HfApi(token=token)
    account = api.whoami()["name"]
    repo_id = args.repo_id or f"{account}/{SPACE_NAME}"
    try:
        existing = api.repo_info(repo_id=repo_id, repo_type="space")
    except Exception:
        existing = None
    if existing is not None and not existing.private:
        raise RuntimeError(f"Le Space {repo_id} existe déjà en public; refuse de publier les annotations dedans.")
    if existing is None:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", private=True)

    with tempfile.TemporaryDirectory(prefix="iris-qwen38-space-") as temporary:
        stage = Path(temporary)
        for name in ("app", "server"):
            shutil.copytree(APP_DIR / name, stage / name)
        for name in ("package.json", "package-lock.json", "nuxt.config.ts", "Dockerfile", "README.md"):
            shutil.copy2(APP_DIR / name, stage / name)
        (stage / "data").mkdir()
        shutil.copy2(ANNOTATIONS, stage / "data/qwen_annotations.jsonl")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(stage),
            commit_message="feat(viewer): deploy private Qwen3.8 browser",
            ignore_patterns=["node_modules/**", ".nuxt/**", ".output/**", ".git/**"],
        )
    api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
    print(f"Space privé déployé : https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
