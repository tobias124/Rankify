#!/usr/bin/env python3
"""
Run Rankify indexers on two JSONL corpora.

Examples
--------
python scripts/run_indexing_on_corpora.py \
  --corpora tests/unit/eu_corpus.jsonl tests/unit/sample.jsonl \
  --indexers bm25 contriever bge colbert dpr \
  --outdir ./indices \
  --device cpu --batch 16 --threads 4
"""

from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys
import time

# ---- Rankify indexers -------------------------------------------------------
from rankify.indexing import (
    LuceneIndexer,
    DPRIndexer,
    ContrieverIndexer,
    ColBERTIndexer,
    BGEIndexer,
    # ANCEIndexer,   # uncomment if you want to run ANCE as well
)

def _ok(pkg: str) -> bool:
    try:
        importlib.import_module(pkg)
        return True
    except Exception:
        return False

def _stem(p: str | Path) -> str:
    return Path(p).stem

def run_bm25(corpus: Path, out_root: Path, threads: int, chunk_size: int,**kwags) -> Path:
    out_dir = out_root / _stem(corpus)
    idx = LuceneIndexer(
        corpus_path=str(corpus),
        output_dir=str(out_dir),
        chunk_size=chunk_size,
        threads=threads,
        index_type="wiki",
        retriever_name="bm25",
    )
    idx.build_index()
    idx.load_index()
    return idx.index_dir

def run_dpr(corpus: Path, out_root: Path, threads: int, batch: int, device: str,**kwags) -> Path:
    if not _ok("pyserini"):
        raise RuntimeError("pyserini is not installed; cannot run DPR indexing.")
    out_dir = out_root / _stem(corpus)
    idx = DPRIndexer(
        corpus_path=str(corpus),
        output_dir=str(out_dir),
        index_type="wiki",
        encoder_name="facebook/dpr-ctx_encoder-single-nq-base",
        batch_size=batch,
        threads=threads,
        device=device,
    )
    idx.build_index()
    idx.load_index()
    return idx.index_dir

def run_contriever(corpus: Path, out_root: Path, batch: int, device: str,**kwags) -> Path:
    out_dir = out_root / _stem(corpus)
    idx = ContrieverIndexer(
        corpus_path=str(corpus),
        output_dir=str(out_dir),
        index_type="wiki",
        encoder_name="facebook/contriever",
        batch_size=batch,              # indexing batch (IDs→FAISS), not HF batch
        embedding_batch_size=max(2, min(32, batch)),  # HF forward batch
        device=device,
    )
    idx.build_index()
    idx.load_index()
    return idx.index_dir

def run_bge(corpus: Path, out_root: Path, batch: int, device: str,**kwags) -> Path:
    out_dir = out_root / _stem(corpus)
    idx = BGEIndexer(
        corpus_path=str(corpus),
        output_dir=str(out_dir),
        index_type="wiki",
        encoder_name="BAAI/bge-large-en-v1.5",
        batch_size=batch,
        device=device,
    )
    idx.build_index()
    idx.load_index()
    return idx.index_dir

def run_colbert(corpus: Path, out_root: Path, batch: int, device: str,**kwags) -> Path:
    # ColBERT needs CUDA toolchain for extensions; if unavailable it may still run
    out_dir = out_root / _stem(corpus)
    idx = ColBERTIndexer(
        corpus_path=str(corpus),
        output_dir=str(out_dir),
        index_type="wiki",
        batch_size=batch,
        device=device,
    )
    idx.build_index()
    idx.load_index()
    return idx.index_dir

# If you want ANCE:
# def run_ance(corpus: Path, out_root: Path, batch: int, device: str) -> Path:
#     out_dir = out_root / _stem(corpus)
#     idx = ANCEIndexer(
#         corpus_path=str(corpus),
#         output_dir=str(out_dir),
#         index_type="wiki",
#         batch_size=batch,
#         device=device,
#     )
#     idx.build_index()
#     idx.load_index()
#     return idx.index_dir


RUNNERS = {
    "bm25": run_bm25,
    "dpr": run_dpr,
    "contriever": run_contriever,
    "bge": run_bge,
    "colbert": run_colbert,
    # "ance": run_ance,
}

def main():
    ap = argparse.ArgumentParser(description="Run Rankify indexers on multiple corpora.")
    ap.add_argument(
        "--corpora", nargs="+", required=True,
        help="Paths to JSONL corpora (e.g., tests/unit/eu_corpus.jsonl tests/unit/sample.jsonl)"
    )
    ap.add_argument(
        "--indexers", nargs="+",
        default=["bm25", "contriever", "bge"],  # safe defaults
        choices=list(RUNNERS.keys()),
        help="Which indexers to run."
    )
    ap.add_argument("--outdir", default="./indices", help="Root directory for all indices.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for dense indexers.")
    ap.add_argument("--batch", type=int, default=32, help="Batch size for dense models.")
    ap.add_argument("--threads", type=int, default=4, help="Threads for BM25/DPR.")
    ap.add_argument("--chunk-size", type=int, default=1024, help="Chunk size for BM25 conversion.")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for corpus in args.corpora:
        corpus = Path(corpus)
        if not corpus.exists():
            print(f"❌ corpus not found: {corpus}", file=sys.stderr)
            continue

        print(f"\n=== Corpus: {corpus} ===")
        for name in args.indexers:
            runner = RUNNERS[name]
            print(f"→ {name} …", end="", flush=True)
            t0 = time.time()
            try:
                idx_dir = runner(
                    corpus=corpus,
                    out_root=out_root,
                    threads=args.threads,
                    chunk_size=args.chunk_size,  # only bm25 uses this
                    batch=args.batch,
                    device=args.device,
                )
                dt = time.time() - t0
                print(f" done in {dt:.1f}s  -> {idx_dir}")
                summary.append((corpus.name, name, str(idx_dir)))
            except Exception as e:
                print(" failed")
                print(f"   {type(e).__name__}: {e}")

    if summary:
        print("\nSummary")
        for c, n, p in summary:
            print(f"  {c:>25}  {n:<11}  {p}")

if __name__ == "__main__":
    main()


"""
# example – adjust paths to your files
python run_indexing_on_corpora.py \
  --corpora data/EU_corpus.jsonl \
  --indexers   bge colbert dpr \
  --outdir ./indices --device cuda --batch 16 --threads 4
#bm25 contriever
cli

# BM25 on both corpora
rankify-index index ./data/sample.jsonl   --retriever bm25 --output ./indicescli

rankify-index index ./data/sample.jsonl   --retriever contriever --device cpu --batch_size 16 --output ./indicescli
rankify-index index ./data/sample.jsonl      --retriever bge --device cpu --batch_size 16 --output ./indicescli

# ColBERT (may require CUDA/toolchain)
rankify-index index ./data/sample.jsonl      --retriever colbert --device cuda --batch_size 32 --output ./indicescli


# dpr

rankify-index index ./data/sample.jsonl \
  --retriever dpr \
  --encoder facebook/dpr-ctx_encoder-single-nq-base \
  --batch_size 16 \
  --device cuda \
  --output ./indicescli


rankify-index index ./sample.jsonl   --retriever bm25 --output ./indicescli  
"""