"""Clone the private Hugging Face dataset, fetch its LFS assets and index images."""

from __future__ import annotations

import base64
import json
import os
import pathlib
import subprocess
import sys
import threading
import time


REPO_ID = "ethanzxv/pixelart-dataset"
REPO_URL = f"https://huggingface.co/datasets/{REPO_ID}"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "pixelart-dataset"
REPO_DIR = DATASET_DIR / ".gitrepo"
STATUS_FILE = DATASET_DIR / ".viewer-download-status.json"
INDEX_FILE = DATASET_DIR / "viewer-index.json"
TOKEN = os.environ.get("HF_ACCESS_TOKEN", "").strip()


def write_status(**values: object) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    status = {"updatedAt": time.time(), **values}
    temporary = STATUS_FILE.with_name(f".viewer-download-status-{os.getpid()}.tmp")
    temporary.write_text(json.dumps(status), encoding="utf-8")
    for attempt in range(10):
        try:
            temporary.replace(STATUS_FILE)
            return
        except PermissionError:
            time.sleep(0.1 * (attempt + 1))
    STATUS_FILE.write_text(json.dumps(status), encoding="utf-8")


def git_environment(skip_smudge: bool) -> dict[str, str]:
    environment = os.environ.copy()
    auth = base64.b64encode(f"user:{TOKEN}".encode("utf-8")).decode("ascii")
    environment["GIT_CONFIG_COUNT"] = "1"
    environment["GIT_CONFIG_KEY_0"] = "http.https://huggingface.co/.extraheader"
    environment["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: Basic {auth}"
    if skip_smudge:
        environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    else:
        environment.pop("GIT_LFS_SKIP_SMUDGE", None)
    return environment


def run(command: list[str], *, environment: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        if "401" in detail or "403" in detail or "Unauthorized" in detail:
            raise RuntimeError("Accès refusé par Hugging Face. Vérifie le jeton et son accès au dépôt.")
        if "Signature=" in detail or "X-Amz-" in detail or "cdn-lfs" in detail:
            raise RuntimeError("Hugging Face a refusé le téléchargement de certains fichiers Git LFS. Vérifie l’accès au dépôt.")
        raise RuntimeError(detail[-2000:] or f"{command[0]} failed with exit code {result.returncode}")


def lfs_counts(environment: dict[str, str]) -> tuple[int, int]:
    result = subprocess.run(
        ["git", "-C", str(REPO_DIR), "lfs", "ls-files", "--long"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        return 0, 0
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    downloaded = sum(1 for line in lines if len(line.split(maxsplit=2)) > 1 and line.split(maxsplit=2)[1] == "*")
    return downloaded, len(lines)


def pull_lfs(environment: dict[str, str]) -> None:
    downloaded, total = lfs_counts(environment)
    if total and downloaded == total:
        write_status(status="downloading", done=total, total=total, message="Images Git LFS déjà à jour")
        return
    write_status(status="downloading", done=downloaded, total=total, message="Téléchargement des images Git LFS")
    run(["git", "-C", str(REPO_DIR), "lfs", "fetch", "origin", "main"], environment=environment)
    run(["git", "-C", str(REPO_DIR), "lfs", "checkout"], environment=environment)
    downloaded, total = lfs_counts(environment)
    if downloaded != total:
        raise RuntimeError(f"Git LFS reported {downloaded} downloaded assets out of {total}.")
    write_status(status="downloading", done=downloaded, total=total, message="Images téléchargées, indexation en cours")


def build_viewer_index() -> int:
    image_files = sorted((REPO_DIR / "data").rglob("*.png"))
    rows: list[dict[str, str]] = []
    total = len(image_files)
    write_status(status="indexing", done=0, total=total, message="Préparation de l'index local")

    for position, image_path in enumerate(image_files, start=1):
        caption_path = image_path.with_suffix(".txt")
        try:
            caption = caption_path.read_text(encoding="utf-8", errors="replace").strip() if caption_path.is_file() else ""
        except OSError:
            caption = ""
        rows.append({
            "path": image_path.relative_to(REPO_DIR).as_posix(),
            "caption": caption,
            "folder": image_path.parent.name,
            "name": image_path.stem,
        })
        if position % 500 == 0:
            write_status(status="indexing", done=position, total=total, message="Préparation de l'index local")

    temporary = INDEX_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    temporary.replace(INDEX_FILE)
    return len(rows)


def main() -> int:
    if not TOKEN:
        print("HF_ACCESS_TOKEN is required", file=sys.stderr)
        return 2

    try:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        if not (REPO_DIR / ".git").exists():
            write_status(status="cloning", done=0, total=0, message="Clonage des fichiers du dépôt Hugging Face")
            run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], environment=git_environment(skip_smudge=True))
        else:
            write_status(status="listing", done=0, total=0, message="Vérification des changements Hugging Face")
            run(["git", "-C", str(REPO_DIR), "fetch", "--no-tags", "origin", "main"], environment=git_environment(skip_smudge=True))
            run(["git", "-C", str(REPO_DIR), "merge", "--ff-only", "FETCH_HEAD"], environment=git_environment(skip_smudge=True))

        environment = git_environment(skip_smudge=False)
        run(["git", "-C", str(REPO_DIR), "config", "lfs.concurrenttransfers", "24"], environment=environment)
        pull_lfs(environment)
        image_count = build_viewer_index()
        write_status(status="complete", done=image_count, total=image_count, images=image_count, message="Dataset téléchargé et indexé")
        print(f"Indexed {image_count} local images", flush=True)
        return 0
    except Exception as error:  # The token is never included in status or log output.
        message = str(error)
        write_status(status="error", done=0, total=0, message=message or "Échec du téléchargement")
        print(f"Dataset download failed: {message}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
