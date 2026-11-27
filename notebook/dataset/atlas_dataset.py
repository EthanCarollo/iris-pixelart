"""Read a pinned dataset checkout and hydrate selected Git LFS files."""

import json
import os
from pathlib import Path, PurePosixPath
import subprocess


ATLAS_URL = "ssh://git@atlas.methil.group:2222/sprite-pixelart-dataset.git"
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"


class AtlasDataset:
    def __init__(self, project_root):
        self.url = os.environ.get("ATLAS_DATASET_SSH", ATLAS_URL)
        configured_dir = os.environ.get("ATLAS_DATASET_DIR")
        cache = project_root / "notebook/dataset/atlas_work/sprite-pixelart-dataset"
        existing = project_root / "notebook/dataset/taxonomy/lfm3_work/atlas_dataset"
        self.root = Path(configured_dir).expanduser().resolve() if configured_dir else cache
        if not configured_dir and not cache.exists() and self._has_atlas_origin(existing):
            self.root = existing

        if not self.root.exists():
            self.root.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth=1", "--branch", "main", "--", self.url, str(self.root)],
                env=self._skip_smudge(), check=True,
            )
        elif not self._has_atlas_origin(self.root):
            raise ValueError(f"Le checkout {self.root} doit avoir {self.url} comme remote origin.")

        self._git("fetch", "--no-tags", "origin", "refs/heads/main")
        self.revision = self._git("rev-parse", "FETCH_HEAD")
        if self._git("rev-parse", "HEAD") != self.revision:
            self._git("merge", "--ff-only", self.revision, env=self._skip_smudge())
            self._check_revision()

    def _has_atlas_origin(self, path):
        if not (path / ".git").exists():
            return False
        result = subprocess.run(
            ["git", "-C", str(path), "remote", "get-url", "origin"],
            capture_output=True, text=True,
        )
        return result.returncode == 0 and result.stdout.strip() == self.url

    @staticmethod
    def _skip_smudge():
        return {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}

    def _git(self, *args, env=None):
        return subprocess.check_output(
            ["git", "-C", str(self.root), *args], env=env, text=True,
        ).strip()

    def _check_revision(self):
        if self._git("rev-parse", "HEAD") != self.revision:
            raise RuntimeError(f"Le checkout doit être sur la révision Atlas {self.revision}.")

    def paths(self, *directories):
        result = self._git("ls-tree", "-r", "-z", "--name-only", self.revision, "--", *directories)
        return result.rstrip("\0").split("\0") if result else []

    def path(self, relative_path):
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in relative_path:
            raise ValueError(f"Chemin hors du dataset : {relative_path}")
        return self.root / relative

    @staticmethod
    def is_pointer(path):
        with path.open("rb") as stream:
            return stream.read(len(LFS_POINTER)) == LFS_POINTER

    def hydrate(self, relative_paths, *, allow_missing=False):
        """Download only the requested pointer files for this revision."""
        self._check_revision()
        files = {name: self.path(name) for name in dict.fromkeys(relative_paths)}
        pointers = [name for name, path in files.items() if path.is_file() and self.is_pointer(path)]
        for start in range(0, len(pointers), 250):
            batch = pointers[start:start + 250]
            # Git LFS uses comma-separated wildmatch patterns.
            patterns = ["/" + "".join("\\" + c if c in "*?[]\\" else c for c in name) for name in batch]
            if any("," in name for name in batch):
                raise ValueError("Un chemin Git LFS contient une virgule.")
            self._git("lfs", "fetch", "--include=" + ",".join(patterns), "--exclude=", "origin", self.revision)
            self._git("lfs", "checkout", "--", *batch)
        available = {name: path for name, path in files.items() if path.is_file() and not self.is_pointer(path)}
        if not allow_missing and len(available) != len(files):
            first = next(name for name in files if name not in available)
            raise FileNotFoundError(f"Fichier indisponible dans Atlas à {self.revision} : {first}")
        return available

    def qwen_annotations(self, override=None, *, exact_palette=False):
        if override:
            return self.hydrate([override])[override]
        candidates = sorted(
            (name for name in self.paths("annotations/")
             if name.startswith(("annotations/checkpoints/", "annotations/dataset_lab_sync/",
                                 "annotations/taxonomy_runs/"))
             and name.endswith(("annotations_v3_4_exact_palette.jsonl",
                                "annotations_v3_4_full_color.jsonl",
                                "annotations_v3_4_color_partial.jsonl", "/annotations.jsonl"))
             and (not exact_palette or name.endswith("_exact_palette.jsonl"))),
            key=lambda name: (name.split("/")[2], name.endswith("_exact_palette.jsonl"), name),
            reverse=True,
        )
        for name in candidates:
            path = self.hydrate([name])[name]
            with path.open(encoding="utf-8") as stream:
                row = next((json.loads(line) for line in stream if line.strip()), {})
            model = str(row.get("model", "")).lower()
            if "qwen3.8" in model and "gyro" in model and "dominant_color_families" in row.get("labels", {}):
                return path
        raise FileNotFoundError("Aucun snapshot Qwen3.8 Gyro-S avec taxonomie couleur dans Atlas.")


def annotation_asset_path(row):
    source = row["source"]
    relative_path = source["relative_path"].replace("\\", "/")
    return f'{source.get("root", "data")}/{relative_path}'
