"""Download the private Hugging Face dataset and prepare a local image index."""

from __future__ import annotations

import concurrent.futures
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


REPO_ID = "ethanzxv/pixelart-dataset"
REPO_TYPE = "datasets"
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets" / "pixelart-dataset"
STATUS_FILE = DATASET_DIR / ".viewer-download-status.json"
INDEX_FILE = DATASET_DIR / "viewer-index.json"
TOKEN = os.environ.get("HF_ACCESS_TOKEN", "").strip()
WORKERS = 24
STATUS_LOCK = threading.Lock()


def write_status(**values: object) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    status = {"updatedAt": time.time(), **values}
    temporary = STATUS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(status), encoding="utf-8")
    temporary.replace(STATUS_FILE)


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Do not forward the private token from Hugging Face to its storage CDN."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        redirected = super().redirect_request(request, file_pointer, code, message, headers, new_url)
        if redirected is not None:
            original_host = urllib.parse.urlparse(request.full_url).hostname
            redirected_host = urllib.parse.urlparse(new_url).hostname
            if original_host != redirected_host:
                redirected.remove_header("Authorization")
                redirected.unredirected_hdrs.pop("Authorization", None)
                redirected.remove_header("authorization")
                redirected.unredirected_hdrs.pop("authorization", None)
        return redirected


OPENER = urllib.request.build_opener(SafeRedirectHandler())


def fetch_repo_files() -> list[str]:
    url = f"https://huggingface.co/api/{REPO_TYPE}/{REPO_ID}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}", "User-Agent": "iris-pixelart-dataset-viewer"})
    with OPENER.open(request, timeout=60) as response:
        repository = json.load(response)
    return [entry["rfilename"] for entry in repository.get("siblings", []) if entry.get("rfilename")]


def download_file(relative_path: str) -> tuple[str, str | None]:
    destination = DATASET_DIR.joinpath(*pathlib.PurePosixPath(relative_path).parts)
    if destination.exists() and destination.is_file():
        return relative_path, None

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    encoded_path = urllib.parse.quote(relative_path, safe="/")
    url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{encoded_path}?download=true"
    headers = {"Authorization": f"Bearer {TOKEN}", "User-Agent": "iris-pixelart-dataset-viewer"}
    error_message = "download failed"

    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers=headers)
            with OPENER.open(request, timeout=180) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            partial.replace(destination)
            return relative_path, None
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as error:
            error_message = f"{type(error).__name__}: {error}"
            try:
                partial.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < 4:
                time.sleep(min(2 ** attempt, 16))

    return relative_path, error_message


def build_viewer_index() -> int:
    image_files = sorted((DATASET_DIR / "data").rglob("*.png"))
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
            "path": image_path.relative_to(DATASET_DIR).as_posix(),
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
        write_status(status="listing", done=0, total=0, message="Lecture du manifeste Hugging Face")
        files = fetch_repo_files()
        if not files:
            raise RuntimeError("Hugging Face returned an empty repository manifest")

        write_status(status="downloading", done=0, total=len(files), message="Téléchargement des fichiers du dataset")
        errors: list[tuple[str, str]] = []
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(download_file, relative_path) for relative_path in files]
            for future in concurrent.futures.as_completed(futures):
                path, error = future.result()
                completed += 1
                if error:
                    errors.append((path, error))
                if completed % 250 == 0 or completed == len(files):
                    write_status(status="downloading", done=completed, total=len(files), errors=len(errors), message="Téléchargement des fichiers du dataset")
                    print(f"{completed}/{len(files)} files", flush=True)

        if errors:
            summary = [{"path": path, "error": message} for path, message in errors[:20]]
            write_status(status="error", done=completed, total=len(files), errors=len(errors), details=summary, message="Le téléchargement est incomplet")
            return 1

        image_count = build_viewer_index()
        write_status(status="complete", done=len(files), total=len(files), images=image_count, message="Dataset téléchargé et indexé")
        print(f"Downloaded {len(files)} files and indexed {image_count} images", flush=True)
        return 0
    except Exception as error:  # Keep the error in the local status file, never the token.
        write_status(status="error", done=0, total=0, message="Échec du téléchargement", details=str(error))
        print(f"Dataset download failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
