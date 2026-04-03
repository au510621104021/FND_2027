"""Helpers for resolving paths across local, Colab, and Kaggle environments."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_colab() -> bool:
    return "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ


def is_kaggle() -> bool:
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/input").exists()


def candidate_roots(project_root: Path) -> list[Path]:
    roots = [project_root.resolve(), Path.cwd().resolve()]

    if is_colab():
        roots.extend([
            Path("/content"),
            Path("/content/drive/MyDrive"),
            Path("/content/drive/Shareddrives"),
        ])
    if is_kaggle():
        roots.extend([
            Path("/kaggle/input"),
            Path("/kaggle/working"),
            Path("/kaggle/temp"),
        ])

    # Preserve order while dropping duplicates.
    deduped = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def resolve_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute() and path.exists():
        return path.resolve()

    for root in candidate_roots(project_root):
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate

        # Common case: only the folder name matters after upload/extract.
        basename_candidate = (root / path.name).resolve()
        if basename_candidate.exists():
            return basename_candidate

    return (project_root / path).resolve()


def resolve_path_list(paths: list[str], project_root: Path) -> list[str]:
    return [str(resolve_path(path_value, project_root)) for path_value in paths]
