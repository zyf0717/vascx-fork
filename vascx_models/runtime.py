from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def configure_runtime_environment() -> None:
    root = repo_root()

    mplconfigdir = Path(os.environ.setdefault("MPLCONFIGDIR", str(root / ".mplconfig")))
    cache_dir = Path(os.environ.setdefault("XDG_CACHE_HOME", str(root / ".cache")))
    model_releases_dir = Path(
        os.environ.setdefault("RTNLS_MODEL_RELEASES", str(root / "model_releases"))
    )

    mplconfigdir.mkdir(exist_ok=True, parents=True)
    cache_dir.mkdir(exist_ok=True, parents=True)
    model_releases_dir.mkdir(exist_ok=True, parents=True)

    for model_path in root.glob("*/*.pt"):
        if model_releases_dir in model_path.parents:
            continue

        symlink_path = model_releases_dir / model_path.name
        if symlink_path.exists() or symlink_path.is_symlink():
            try:
                if symlink_path.resolve() == model_path.resolve():
                    continue
            except FileNotFoundError:
                pass
            symlink_path.unlink()

        symlink_path.symlink_to(model_path.resolve())
