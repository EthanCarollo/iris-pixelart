#!/usr/bin/env python3
"""New, independent Mirage annotations. One image by default; --all for the corpus."""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import time

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
TAXONOMY = ROOT / "notebook/dataset/taxonomy"
WORK = TAXONOMY / "mirage_work"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
PIPELINE_VERSION = 1


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read_jsonl(path):
    if not path.exists():
        return []
    # Refuse a truncated tail rather than appending a second JSON object to it.
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Dernière ligne incomplète dans {path}; la sauvegarder puis la retirer avant reprise.")
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def taxonomy_schema():
    dimensions, paths, hashes = {}, {}, {}
    def leaves(nodes, prefix=()):
        result = {}
        for name, node in nodes.items():
            branch = (*prefix, name)
            if node["children"]:
                result.update(leaves(node["children"], branch))
            else:
                result[name] = list(branch)
        return result
    for name in ("taxonomies.json", "taxonomies_couleurs.json"):
        data = (TAXONOMY / name).read_bytes()
        hashes[name] = sha(data)
        doc = json.loads(data)
        dimensions.update(doc["dimensions"])
        paths.update({key: leaves(tree) for key, tree in doc["hierarchies"].items()})
    return dimensions, paths, hashes


def make_form(dimensions, paths):
    form = {}
    for field, options in paths.items():
        # Exact source measurements do not need a prediction.
        if field in {"palette_size", "background_type"}:
            continue
        if dimensions[field].get("cardinality") == "multi_label":
            # Native Mirage forms support enums, not JSON-schema arrays.
            for color in options:
                if color not in {"unknown", "not_applicable"}:
                    form[f"dominant_color_{color}"] = {"enum": ["present", "absent", "unknown"]}
        else:
            form[field] = {"enum": list(options)}
    form["background_type"] = {"enum": ["solid_color", "complex", "unknown"]}
    form["caption_candidate"] = {"type": "string"}
    return form


def prepare_image(path, background, upscale):
    with Image.open(path) as source:
        frames = getattr(source, "n_frames", 1)
        rgba = source.convert("RGBA")
    colors = Counter(rgba.get_flattened_data() if hasattr(rgba, 'get_flattened_data') else rgba.getdata())
    visible = Counter()
    for color, count in colors.items():
        if color[3] > 0:
            visible[color[:3]] += count
    count = len(visible)
    palette_size = next((name for bound, name in [(0, "not_applicable"), (4, "very_limited"),
                        (8, "limited"), (16, "moderate"), (32, "broad")]
                        if count <= bound), "very_broad")
    alpha = rgba.getchannel("A")
    scale = max(1, min(8, 512 // max(rgba.size))) if upscale else 1
    resized = rgba.resize((rgba.width * scale, rgba.height * scale), Image.Resampling.NEAREST)
    canvas = Image.new("RGBA", resized.size, background)
    canvas.alpha_composite(resized)
    rgb = canvas.convert("RGB")
    measurements = {
        "width": rgba.width, "height": rgba.height, "frame_count": frames,
        "analyzed_frame": 0, "has_transparency": alpha.getextrema()[0] < 255,
        "visible_rgb_color_count": count, "palette_size": palette_size,
        "visible_bbox": alpha.getbbox(),
        "transparent_pixels": sum(n for rgba_value, n in colors.items() if rgba_value[3] == 0),
        "partially_transparent_pixels": sum(n for rgba_value, n in colors.items() if 0 < rgba_value[3] < 255),
        "palette_hex": [{"hex": "#%02X%02X%02X" % c, "pixel_count": n}
                        for c, n in visible.most_common()],
        "model_input": {"size": list(rgb.size), "nearest_neighbor_scale": scale,
                        "background": background, "rgb_sha256": sha(rgb.tobytes())},
    }
    return rgb, measurements


def normalize(extraction, dimensions, paths, measurements, threshold):
    raw = extraction.get("form", {})
    labels, warnings = {}, []
    def choice(field, allowed):
        result = raw.get(field, {})
        value = result.get("value")
        probability = result.get("probabilities", {}).get(value, result.get("probability", 0))
        if value not in allowed or probability < threshold:
            warnings.append(f"{field}: réponse absente/invalide ou probabilité < {threshold}")
            return "unknown"
        return value
    for field, allowed in paths.items():
        if field == "palette_size":
            labels[field] = measurements["palette_size"]
        elif field == "background_type" and measurements["has_transparency"]:
            labels[field] = "transparent"
        elif dimensions[field].get("cardinality") == "multi_label":
            colors = [color for color in allowed if color not in {"unknown", "not_applicable"}
                      and choice(f"dominant_color_{color}", {"present", "absent", "unknown"}) == "present"]
            labels[field] = colors or ["unknown"]
        else:
            labels[field] = choice(field, allowed)
    if labels["background_type"] in {"transparent", "solid_color"}:
        if labels["background_scene"] != "not_applicable":
            warnings.append("background_scene: normalisé en not_applicable d'après le fond")
        labels["background_scene"] = "not_applicable"
    elif labels["background_type"] == "complex" and labels["background_scene"] == "not_applicable":
        labels["background_scene"] = "unknown"
    caption = raw.get("caption_candidate", {}).get("value")
    if not isinstance(caption, str) or caption.strip().lower() in {"", "-", "unknown", "n/a", "none"}:
        caption = None
        warnings.append("Mirage n'a pas produit de caption exploitable")
    return labels, caption, warnings


def annotate(model, path, source, dimensions, paths, args, config_id):
    image, measured = prepare_image(path, args.background, not args.native_size)
    started = time.monotonic()
    # Neither old captions, source filenames nor Qwen labels are passed to Mirage.
    description = model.describe(image, language="en")
    extraction = model.extract(image, make_form(dimensions, paths), language="en")
    elapsed = time.monotonic() - started
    labels, caption, warnings = normalize(extraction, dimensions, paths, measured, args.min_probability)
    key = source["root"] + "/" + source["relative_path"]
    return {
        "asset_id": sha(key.encode())[:24], "source": source,
        "source_image_sha256": sha(path.read_bytes()), "run_config_id": config_id,
        "model": "Cactus-Compute/mirage", "pipeline": "mirage_independent_v1",
        "labels": labels,
        "taxonomy_paths": {key: [paths[key][v] for v in value] if isinstance(value, list)
                           else paths[key].get(value, []) for key, value in labels.items()},
        "label_provenance": {key: "source_pixels" if key == "palette_size" or
                             (key == "background_type" and measured["has_transparency"])
                             else "mirage_form_normalized" for key in labels},
        "caption_candidate": caption,
        "caption_status": "candidate_unreviewed" if caption else "missing",
        "mirage_description": description, "mirage_extraction": extraction,
        "measurements": measured, "warnings": warnings,
        "need_rework": None if labels["visual_quality"] == "unknown" else labels["visual_quality"] == "needs_rework",
        "review_status": "auto_labeled", "needs_taxonomy_review": True,
        "rights_status": "unknown", "inference_seconds": elapsed,
        "annotated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, image


def report(output, records, inventory_count):
    counts = {field: dict(Counter(value for row in records for value in
              (row["labels"][field] if isinstance(row["labels"][field], list) else [row["labels"][field]])))
              for field in records[0]["labels"]} if records else {}
    summary = {"annotated": len(records), "inventory_images": inventory_count,
               "caption_present": sum(bool(r["caption_candidate"]) for r in records),
               "requires_human_review": len(records), "label_distributions": counts,
               "mean_inference_seconds": sum(r["inference_seconds"] for r in records) / len(records) if records else None}
    write_json(output / "summary.json", summary)
    cards = []
    escape = lambda value: html.escape(str(value))
    for row in records:
        preview = output / "previews" / (row["asset_id"] + ".png")
        uri = "data:image/png;base64," + base64.b64encode(preview.read_bytes()).decode() if preview.exists() else ""
        labels = "".join(f"<tr><th>{escape(k)}</th><td>{escape(v)}</td></tr>" for k, v in row["labels"].items())
        raw = json.dumps({"describe": row["mirage_description"], "extract": row["mirage_extraction"]}, ensure_ascii=False, indent=2)
        cards.append(f'<article><h2>{escape(row["source"]["relative_path"])}</h2>'
                     f'<img loading="lazy" src="{uri}" alt="Image transmise à Mirage">'
                     f'<p><b>Caption Mirage :</b> {escape(row["caption_candidate"] or "Aucune caption exploitable")}</p>'
                     f'<table>{labels}</table><p>{escape("; ".join(row["warnings"]))}</p>'
                     f'<details><summary>Sorties Mirage brutes</summary><pre>{escape(raw)}</pre></details></article>')
    (output / "review.html").write_text('<!doctype html><html lang="fr"><meta charset="utf-8">'
        '<title>Nouvelles annotations Mirage</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:16px;background:#f2f4f7;color:#192231}'
        'article{background:white;padding:24px;border-radius:16px;margin:24px 0;overflow:auto}img{image-rendering:pixelated;max-width:100%;max-height:512px;float:right;margin:0 0 16px 20px}'
        'h2{font-size:18px;overflow-wrap:anywhere}th{text-align:left}td,th{padding:4px 12px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{clear:both}</style>'
        f'<h1>Nouvelles annotations Mirage</h1><p>{len(records)} images annotées / {inventory_count} images sélectionnées. '
        'Aucune annotation Qwen réutilisée. Les probabilités du modèle ne mesurent pas sa justesse visuelle. '
        'Vérifier sujet, vue, pose, couleurs, objets inventés, omissions et caption avant entraînement.</p>'
        + ''.join(cards) + '</html>', encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="ethanzxv/pixelart-dataset")
    parser.add_argument("--revision", default="main", help="Résolu en commit immuable au premier lancement")
    parser.add_argument("--image", help="Chemin local ou chemin HF data/... ou source_sheets/...")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Toutes les images, reprise automatique")
    selection.add_argument("--limit", type=int, default=None, help="Nombre total à atteindre (défaut : 1)")
    parser.add_argument("--data-only", action="store_true", help="Exclure source_sheets/")
    parser.add_argument("--output", type=Path, default=WORK)
    parser.add_argument("--background", default="white")
    parser.add_argument("--native-size", action="store_true", help="Pas d'agrandissement nearest-neighbor")
    parser.add_argument("--min-probability", type=float, default=0.6)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit doit être positif")
    if not 0 <= args.min_probability <= 1:
        parser.error("--min-probability doit être compris entre 0 et 1")
    if args.image and (args.all or args.limit is not None):
        parser.error("--image s'utilise seul, sans --all/--limit")
    load_dotenv(ROOT / "dataset-viewer/.env", override=False)
    os.environ.setdefault("CACTUS_TELEMETRY", "0")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_ACCESS_TOKEN")
    api = HfApi(token=token)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "run_manifest.json"
    previous = json.loads(config_path.read_text()) if config_path.exists() else None
    local_image = Path(args.image).resolve() if args.image and Path(args.image).is_file() else None
    if local_image:
        revision = None
        inventory = [str(local_image)]
    else:
        revision = previous["dataset_revision"] if previous and args.revision == "main" else api.repo_info(args.repo, repo_type="dataset", revision=args.revision).sha
        inventory_path = args.output / "inventory.json"
        inventory_doc = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
        if inventory_doc.get("revision") != revision or inventory_doc.get("repo") != args.repo:
            files = [entry.rfilename for entry in api.repo_info(
                args.repo, repo_type="dataset", revision=revision).siblings]
            inventory_doc = {"repo": args.repo, "revision": revision, "paths": sorted(p for p in files
                if p.startswith(("data/", "source_sheets/")) and PurePosixPath(p).suffix.lower() in EXTENSIONS)}
            write_json(inventory_path, inventory_doc)
        inventory = inventory_doc["paths"]
        if args.image:
            if args.image not in inventory:
                parser.error("Image absente de l'inventaire à cette révision")
            inventory = [args.image]
        elif args.data_only:
            inventory = [p for p in inventory if p.startswith("data/")]
    dimensions, paths, hashes = taxonomy_schema()
    import cactus
    model = cactus.Mirage()
    config = {"pipeline_version": PIPELINE_VERSION, "repo": args.repo, "dataset_revision": revision,
              "taxonomy_sha256": hashes, "cactus_compute_version": importlib.metadata.version("cactus-compute"),
              "model_weights_sha256": sha(Path(model.weights).read_bytes()),
              "model_weights_cache": str(model.weights),
              "background": args.background, "native_size": args.native_size,
              "min_probability": args.min_probability, "language": "en", "form": make_form(dimensions, paths)}
    config_id = sha(json.dumps(config, sort_keys=True).encode())
    if previous and previous["config_id"] != config_id:
        raise SystemExit("Configuration différente : choisir un nouveau --output pour ne pas mélanger les annotations.")
    write_json(config_path, {**config, "config_id": config_id})
    output_file = args.output / "annotations_mirage.jsonl"
    records = read_jsonl(output_file)
    if any(row["run_config_id"] != config_id for row in records):
        raise SystemExit("Le JSONL contient des résultats provenant d'une autre configuration.")
    completed = {(r["source"]["root"], r["source"]["relative_path"]) for r in records}
    if len(completed) != len(records):
        raise SystemExit("Doublons détectés dans le JSONL Mirage.")
    selected = inventory if args.all else inventory[:args.limit or 1]
    (args.output / "previews").mkdir(exist_ok=True)
    errors = 0
    try:
        for position, name in enumerate(selected, 1):
            source = {"root": "local", "relative_path": str(local_image)} if local_image else dict(zip(("root", "relative_path"), name.split("/", 1)))
            if (source["root"], source["relative_path"]) in completed:
                continue
            try:
                path = local_image or Path(hf_hub_download(args.repo, name, repo_type="dataset", revision=revision, token=token))
                row, preview = annotate(model, path, source, dimensions, paths, args, config_id)
                preview.save(args.output / "previews" / (row["asset_id"] + ".png"))
                with output_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                records.append(row)
                completed.add((source["root"], source["relative_path"]))
                print(f"{position}/{len(selected)} · {name} · {row['inference_seconds']:.2f}s · caption={bool(row['caption_candidate'])}", flush=True)
                if len(records) % 100 == 0:
                    report(args.output, records, len(inventory))
            except (OSError, RuntimeError, ValueError) as exc:
                errors += 1
                # Do not serialize HTTP request objects or credentials in errors.
                with (args.output / "errors.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"source": source, "error_type": type(exc).__name__}) + "\n")
                print(f"Échec {name}: {type(exc).__name__}; réessayé au prochain lancement", flush=True)
    finally:
        summary = report(args.output, records, len(inventory))
        print(json.dumps({k: v for k, v in summary.items() if k != "label_distributions"}, ensure_ascii=False))
        print(f"Résultats : {output_file}\nRevue : {args.output / 'review.html'}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
