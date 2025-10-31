import time
import psutil
import os
import json
import csv
from pathlib import Path
import argparse

from rankify.indexing import format_converters, LuceneIndexer


# --- Helpers ---
def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6  # MB

def log_stage(stage_name, start_time, results, index_dir=None, meta=None):
    elapsed = time.time() - start_time
    mem = get_memory_mb()
    size = None
    if index_dir and Path(index_dir).exists():
        size = sum(f.stat().st_size for f in Path(index_dir).rglob("*")) / 1e6

    row = {
        "stage": stage_name,
        "time_s": round(elapsed, 2),
        "memory_mb": round(mem, 2),
        "index_size_mb": round(size, 2) if size else None
    }
    if meta:
        row.update(meta)

    results.append(row)
    print(f"[{stage_name}] Time: {elapsed:.2f}s | Mem: {mem:.2f}MB | IndexSize: {size:.2f}MB")
    return time.time()


def benchmark_indexing(cfg, chunk_size, threads_override, run_dir):
    ts_meta = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    ts_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    results = []

    corpus_path = Path(cfg["corpus_path"])
    index_dir = Path(cfg["index_dir"])
    retriever = cfg.get("retriever", "").lower()

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    if not index_dir.exists():
        index_dir.mkdir(parents=True, exist_ok=True)

    threads = threads_override if threads_override else cfg.get("threads", 8)

    # Meta
    meta = {
        "name": cfg.get("name", "unknown"),
        "retriever": retriever,
        "index_type": cfg.get("index_type", "unknown"),
        "corpus_path": str(corpus_path),
        "corpus_size": format_converters.count_file_lines(corpus_path),
        "chunk_size": chunk_size,
        "threads": threads,
        "timestamp": ts_meta
    }

    print(f"=== Benchmarking {retriever.upper()} | chunk={chunk_size}, threads={threads} ===")

    # --- Benchmark Schritte ---
    if retriever == "bm25":
        # Step 1: Chunking
        t0 = time.time()
        format_converters.chunk_corpus(
            input_path=corpus_path,
            output_dir=index_dir,
            chunk_size=chunk_size,
            threads=threads,
            dense_index=False
        )
        t0 = log_stage("chunk_corpus", t0, results, index_dir, meta=meta)

        # Step 2: Lucene Index build
        indexer = LuceneIndexer(
            corpus_path=corpus_path,
            output_dir=index_dir,
            chunk_size=chunk_size,
            threads=threads,
            index_type=cfg.get("index_type", "wiki"),
            retriever_name="bm25"
        )
        t0 = time.time()
        indexer.build_index()
        t0 = log_stage("lucene_build", t0, results, index_dir, meta=meta)

    else:
        raise NotImplementedError(f"Retriever {retriever} not supported yet.")

    # --- Logs speichern in run_dir ---
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{cfg['name']}_chunk{chunk_size}_threads{threads}_{ts_file}.json"

    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(log_path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Benchmark completed. Logs saved to {log_path}")
    return results

if __name__ == "__main__":
    from datasets import load_dataset
    #~/.cache/huggingface/datasets/
# Natural Questions
    nq = load_dataset("nq_open")

# TriviaQA
    trivia = load_dataset("trivia_qa", "rc")

# WebQuestions
    webq = load_dataset("web_questions")
    exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the benchmark configuration JSON file.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    # Base run dir
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = Path(cfg["benchmark"]["log_file"]).with_suffix("") / ts

    chunk_sizes = cfg.get("chunking", {}).get("chunk_sizes", [cfg["chunking"]["chunk_size"]])
    thread_variants = cfg.get("thread_variants", [cfg.get("threads", 8)])

    for c in chunk_sizes:
        for t in thread_variants:
            benchmark_indexing(cfg, chunk_size=c, threads_override=t, run_dir=run_dir)

    # --- Master CSV für diesen Run erzeugen ---
    import pandas as pd
    all_csv = list(run_dir.glob("*.csv"))
    dfs = [pd.read_csv(f) for f in all_csv]
    master = pd.concat(dfs, ignore_index=True)
    master.to_csv(run_dir / "master_benchmarks.csv", index=False)
    print(f"📊 Master CSV saved at {run_dir/'master_benchmarks.csv'}")




