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

def data_dir() -> str:
    """Return the data dir path (create if missing)."""
    DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR_PATH)
