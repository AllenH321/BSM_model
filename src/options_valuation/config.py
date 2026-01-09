from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

import os

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR_PATH = PROJECT_ROOT / "data"

def load_env(dotenv_path: Path | None = None):
    """Load environment variables from .env at project root."""
    if dotenv_path is None:
        dotenv_path = PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path, override=False)

def get_key(name: str, default=None, required: bool = False):
    """Read an environment variable."""
    val = os.getenv(name, default)
    if required and val is None:
        raise RuntimeError(f"Missing environment variable: {name}")
    return val

def get_project_root(anchor: str = "data") -> Path:
    """
    Find project root by walking up from this file until we find an 'anchor' folder.
    Default anchor='data' works well for your repo layout.
    """
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / anchor).exists():
            return p
    # fallback: repo root = 2 levels above this file (safe fallback)
    return here.parents[1]

PROJECT_ROOT: Path = get_project_root()


def data_dir() -> Path:
    return PROJECT_ROOT / "data"


def raw_dir() -> Path:
    return data_dir() / "raw"


def processed_dir() -> Path:
    return data_dir() / "processed"


def docs_dir() -> Path:
    return PROJECT_ROOT / "docs"