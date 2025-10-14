# rankify/__init__.py
import os
from pathlib import Path

DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "rankify")
os.environ.setdefault("RERANKING_CACHE_DIR", DEFAULT_CACHE_DIR)
Path(os.environ["RERANKING_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)