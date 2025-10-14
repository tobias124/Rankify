import os
from pathlib import Path
DEFAULT_CACHE_DIR = str(Path.home() / ".cache" / "rankify")
os.environ.setdefault("RERANKING_CACHE_DIR", DEFAULT_CACHE_DIR)  # ← Sets variable
Path(os.environ["RERANKING_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

# BUT if download.py has this at the TOP of the file:
# rankify/utils/dataset/download.py
cache_path = os.environ['RERANKING_CACHE_DIR']  # ← FAILS! Variable not set yet