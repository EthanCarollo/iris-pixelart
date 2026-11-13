"""Measure exact, alpha-aware source-image palettes for Qwen rows or the whole dataset."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "notebook/dataset/taxonomy/taxonomy_work/annotations_v3_4_full_color.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "notebook/dataset/taxonomy/taxonomy_work/annotations_v3_4_exact_palette.jsonl"
DEFAULT_DATASET_PALETTE_OUTPUT = PROJECT_ROOT / "notebook/dataset/taxonomy/taxonomy_work/dataset_color_palettes.jsonl.gz"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "dataset-viewer/datasets/pixelart-dataset/.gitrepo/data"
DEFAULT_METADATA = PROJECT_ROOT / "dataset-viewer/datasets/pixelart-dataset/.gitrepo/metadata.jsonl"
DATASET_REPO = "ethanzxv/pixelart-dataset"
CANVAS_MULTIPLE = 16


def load_token() -> str | bool:
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
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or True


def color_record(rgba: tuple[int, int, int, int], count: int, total: int) -> dict[str, Any]:
    red, green, blue, alpha = rgba
    return {
        "hex": f"#{red:02X}{green:02X}{blue:02X}",
        "alpha": alpha,
        "rgba_hex": f"#{red:02X}{green:02X}{blue:02X}{alpha:02X}",
        "pixel_count": count,
        "visible_pixel_fraction": round(count / total, 8) if total else 0.0,
    }


def composite_on_white(rgba: tuple[int, int, int, int]) -> tuple[int, int, int]:
    red, green, blue, alpha = rgba
    return tuple(round(channel * alpha / 255 + 255 * (255 - alpha) / 255) for channel in (red, green, blue))


def exact_palette(image_path: Path) -> dict[str, Any]:
    with Image.open(image_path) as original:
        rgba_image = original.convert("RGBA")
        width, height = rgba_image.size
        raw = rgba_image.tobytes()

    rgba_counts = Counter(zip(raw[0::4], raw[1::4], raw[2::4], raw[3::4]))
    visible_counts = {rgba: count for rgba, count in rgba_counts.items() if rgba[3] > 0}
    visible_count = sum(visible_counts.values())
    rgba_palette = [
        color_record(rgba, count, visible_count)
        for rgba, count in sorted(visible_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    # The training notebook composites sprites onto white and pads them to multiples of 16.
    composed_counts: Counter[tuple[int, int, int]] = Counter()
    for rgba, count in rgba_counts.items():
        composed_counts[composite_on_white(rgba)] += count
    canvas_width = math.ceil(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    canvas_height = math.ceil(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE
    padding_count = canvas_width * canvas_height - width * height
    if padding_count:
        composed_counts[(255, 255, 255)] += padding_count
    canvas_pixels = canvas_width * canvas_height
    training_palette = [
        {
            "hex": f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}",
            "pixel_count": count,
            "canvas_pixel_fraction": round(count / canvas_pixels, 8),
        }
        for rgb, count in sorted(composed_counts.items(), key=lambda item: (-item[1], item[0]))
    ]

    return {
        "method": "exact_source_rgba_frequency",
        "source_size": [width, height],
        "visible_pixel_count": visible_count,
        "transparent_pixel_count": sum(count for rgba, count in rgba_counts.items() if rgba[3] == 0),
        "partially_transparent_pixel_count": sum(count for rgba, count in rgba_counts.items() if 0 < rgba[3] < 255),
        "distinct_rgb_color_count": len({rgba[:3] for rgba in visible_counts}),
        "distinct_rgba_color_count": len(visible_counts),
        "palette_hex": rgba_palette,
        "training_canvas_16": {
            "size": [canvas_width, canvas_height],
            "added_white_padding_pixel_count": padding_count,
            "palette_hex": training_palette,
        },
    }


def read_complete_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    snapshot_size = path.stat().st_size
    with path.open("rb") as stream:
        while stream.tell() < snapshot_size:
            line = stream.readline(snapshot_size - stream.tell())
            if not line.endswith(b"\n"):
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cached_or_downloaded_image(relative_path: str, token: str | bool) -> Path:
    filename = f"data/{relative_path.replace(chr(92), '/')}"
    try:
        return Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=filename,
                                    token=token, local_files_only=True))
    except Exception:
        return Path(hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=filename, token=token))


def measure_row(index: int, row: dict[str, Any], image_root: Path, token: str | bool) -> tuple[int, dict[str, Any], str]:
    source = row.get("source", {})
    relative_path = source.get("relative_path")
    root = source.get("root", "data")
    if not relative_path:
        row["color_measurements"] = {"method": "exact_source_rgba_frequency", "status": "missing_source_path"}
        return index, row, "missing"

    local_path = image_root / (Path(relative_path) if root == "data" else Path(root) / str(relative_path))
    try:
        try:
            measurements = exact_palette(local_path)
        except (OSError, UnidentifiedImageError, ValueError):
            # The existing Git checkout may contain a 129-byte Git LFS pointer rather than PNG bytes.
            downloaded = cached_or_downloaded_image(str(relative_path), token)
            measurements = exact_palette(downloaded)
        row["color_measurements"] = {"status": "measured", **measurements}
        return index, row, "measured"
    except FileNotFoundError:
        row["color_measurements"] = {"method": "exact_source_rgba_frequency", "status": "missing_image"}
        return index, row, "missing"
    except Exception as error:
        row["color_measurements"] = {
            "method": "exact_source_rgba_frequency",
            "status": "unavailable",
            "error_type": type(error).__name__,
        }
        return index, row, "unavailable"


def measure_dataset_asset(
    index: int, row: dict[str, Any], image_root: Path, token: str | bool
) -> tuple[int, dict[str, Any], str]:
    image = str(row.get("image") or row.get("file_name") or "")
    marker = "/api/sprites/"
    relative_path = image.split(marker, 1)[1] if marker in image else image.lstrip("/")
    if relative_path.startswith("data/"):
        relative_path = relative_path[5:]
    result: dict[str, Any] = {
        "asset_id": row.get("id"),
        "source": {"root": "data", "relative_path": relative_path},
    }
    if not relative_path:
        result["color_measurements"] = {
            "method": "exact_source_rgba_frequency", "status": "missing_source_path"
        }
        return index, result, "missing"

    local_path = image_root / relative_path
    try:
        try:
            measurements = exact_palette(local_path)
        except (OSError, UnidentifiedImageError, ValueError):
            # Git LFS checkouts can contain a pointer instead of the actual image bytes.
            downloaded = cached_or_downloaded_image(str(relative_path), token)
            measurements = exact_palette(downloaded)
        result["color_measurements"] = {"status": "measured", **measurements}
        return index, result, "measured"
    except FileNotFoundError:
        result["color_measurements"] = {
            "method": "exact_source_rgba_frequency", "status": "missing_image"
        }
        return index, result, "missing"
    except Exception as error:
        result["color_measurements"] = {
            "method": "exact_source_rgba_frequency",
            "status": "unavailable",
            "error_type": type(error).__name__,
        }
        return index, result, "unavailable"


def dataset_metadata_path(requested: Path | None, token: str | bool) -> Path:
    if requested and requested.is_file():
        return requested
    if DEFAULT_METADATA.is_file():
        return DEFAULT_METADATA
    return Path(hf_hub_download(
        repo_id=DATASET_REPO, repo_type="dataset", filename="metadata.jsonl", token=token
    ))


def write_dataset_palette_index(
    metadata_path: Path, output_path: Path, image_root: Path, token: str | bool, workers: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    completed = 0
    index = 0
    max_pending = max(1, workers) * 3
    with tempfile.NamedTemporaryFile(dir=output_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with gzip.open(temporary_path, "wt", encoding="utf-8", compresslevel=6) as output_stream:
                with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                    pending = set()
                    with metadata_path.open(encoding="utf-8") as metadata_stream:
                        for line in metadata_stream:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            pending.add(executor.submit(measure_dataset_asset, index, row, image_root, token))
                            index += 1
                            if len(pending) >= max_pending:
                                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                                for future in done:
                                    _, result, status = future.result()
                                    output_stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                                    counts[status] += 1
                                    completed += 1
                                if completed % 1000 < len(done):
                                    print(f"Images traitées : {completed:,} / {index:,} chargées", flush=True)
                    for future in as_completed(pending):
                        _, result, status = future.result()
                        output_stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                        counts[status] += 1
                        completed += 1
                        if completed % 1000 == 0:
                            print(f"Images traitées : {completed:,} / {index:,}", flush=True)
            with temporary_path.open("rb") as compressed_stream:
                os.fsync(compressed_stream.fileno())
            temporary_path.replace(output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    print(f"Images du dataset : {completed:,}")
    print(f"Palettes exactes : {counts['measured']:,} · absentes : {counts['missing']:,} · indisponibles : {counts['unavailable']:,}")
    print(f"Index JSONL : {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all-dataset-assets", action="store_true",
        help="mesurer toutes les images du metadata.jsonl du dataset, sans dépendre de Qwen",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metadata", type=Path, help="metadata.jsonl local; sinon cache HF ou checkout local")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    token = load_token()
    if args.all_dataset_assets:
        output = args.output or DEFAULT_DATASET_PALETTE_OUTPUT
        metadata_path = dataset_metadata_path(args.metadata, token)
        write_dataset_palette_index(
            metadata_path, output, args.image_root, token, args.workers
        )
        return

    output = args.output or DEFAULT_OUTPUT
    rows = read_complete_jsonl(args.input)
    if not rows:
        raise RuntimeError(f"Aucune ligne complète à traiter dans {args.input}")

    results: list[dict[str, Any] | None] = [None] * len(rows)
    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(measure_row, index, row, args.image_root, token) for index, row in enumerate(rows)]
        for completed, future in enumerate(as_completed(futures), start=1):
            index, row, status = future.result()
            results[index] = row
            counts[status] += 1
            if completed % 500 == 0:
                print(f"Images traitées : {completed:,} / {len(rows):,}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            for row in results:
                temporary.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path.replace(output)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    print(f"Annotations : {len(rows):,}")
    print(f"Palettes exactes : {counts['measured']:,} · absentes : {counts['missing']:,} · indisponibles : {counts['unavailable']:,}")
    print(f"JSONL dérivé : {output}")


if __name__ == "__main__":
    main()
