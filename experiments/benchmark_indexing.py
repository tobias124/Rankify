import time
import psutil
import os
import json
import csv
import statistics
from pathlib import Path
import argparse
import pandas as pd

from rankify.indexing import format_converters, LuceneIndexer, DPRIndexer, BGEIndexer, ContrieverIndexer
from analyze_benchmarks import generate_plots

def run_bge_benchmark(cfg, batch_size_variants):
    print("🔧 SKip -- Converting corpus to DPR TSV ...")
    #format_converters.to_tsv(input_path=cfg["corpus_path"],output_dir=Path(cfg["index_dir"]), chunk_size=5000000, threads=32)
    print("✅ Skip -- Conversion complete.\n")
    run_benchmark_variants(cfg, batch_size_variants, "batch_size")

def run_dpr_benchmark(cfg, batch_size_variants):
    print("🔧 Converting to_pyserini_jsonl_dense ...")
    format_converters.to_pyserini_jsonl_dense(input_path=cfg["corpus_path"], output_dir=Path(cfg["index_dir"]))
    print("✅ Conversion complete.\n")
    run_benchmark_variants(cfg, batch_size_variants, "batch_size")

def run_lucene_benchmark(cfg, thread_variants):
    print("🔧 Converting corpus to Pyserini JSONL ...")
    format_converters.to_pyserini_jsonl(input_path=cfg["corpus_path"],output_dir=Path(cfg["index_dir"]))
    print("✅ Conversion complete.\n")
    run_benchmark_variants(cfg, thread_variants)


def run_benchmark_variants(cfg, variants_param, method="threads"):
    """Run benchmarks for each variant in the provided parameter list.
    Args: variants_param (list): List of parameter variants (e.g., thread counts).
          cfg (dict): Benchmark configuration dictionary.
        """
    # --- LOOP OVER THREAD VARIANTS ---
    for t in variants_param:
        print(f"\n=== 🧪 Benchmarking {method}={t} for {args.repeats} repeats ===")

        repeated_runs = []

        for r in range(args.repeats):
            print(f"  → Repeat {r+1}/{args.repeats}")
            result = benchmark_indexing(cfg, override_value=t, run_dir=run_dir, method=method)
            repeated_runs.append(result)

        # Aggregate after all repeats for THIS thread value
        summary = aggregate_repeats(repeated_runs)
        summary_path = run_dir / f"{cfg['name']}_{method}{t}_summary.csv"
        pd.DataFrame(summary).to_csv(summary_path, index=False)
        print(f"📊 Summary saved at {summary_path}")


# --- Helpers ---
def validate_config(cfg):
    """Ensure the config contains all required keys."""
    required = ["name", "corpus_path", "index_dir", "retriever", "benchmark"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing required config key: '{key}'")
    if cfg["retriever"].lower() in ["bm25"]:
        if "thread_variants" not in cfg:
            raise ValueError("Config must include 'thread_variants' for BM25 retrievers.")


import psutil

def get_java_memory_mb():
    """Return memory usage of the Java Lucene/Pyserini process in MB."""
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        name = (p.info['name'] or "").lower()
        cmd = " ".join(p.info['cmdline'] or []).lower()

        # detect Java process
        if "java" in name or "java" in cmd:
            try:
                return p.memory_info().rss / 1e6
            except psutil.NoSuchProcess:
                continue

    return None

def get_memory_mb():
    """Return current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6


def log_stage(stage_name, start_time, results, index_dir=None, meta=None, lucene=False):
    """Log time, memory, and index size for a benchmark stage."""
    elapsed = time.time() - start_time
    if lucene:
        mem = get_java_memory_mb()
    else:
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


def benchmark_indexing(cfg, override_value, run_dir, method):
    """Run benchmark indexing for selected retriever."""
    ts_file = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    results = []
    corpus_path = Path(cfg["corpus_path"])
    index_dir = Path(cfg["index_dir"])
    retriever = cfg.get("retriever", "").lower()

    # TODO: check if not already validated
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    if not retriever:
        raise ValueError("Retriever not specified in config.")
    index_dir.mkdir(parents=True, exist_ok=True)

    threads = override_value if override_value else cfg.get("threads", 8)
    ts_meta = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    meta = {
        "name": cfg.get("name", "unknown"),
        "retriever": retriever,
        "index_type": cfg.get("index_type", "unknown"),
        "corpus_path": str(corpus_path),
        "corpus_size": format_converters.count_file_lines(corpus_path),
        "threads": threads,
        "batch_size": override_value if method == "batch_size" else "N/A",
        "timestamp": ts_meta
    }

    print(f"\n=== Benchmarking {retriever.upper()} | {method}={override_value} ===")

    if retriever == "bm25":
        # Lucene Index build
        indexer = LuceneIndexer(
            corpus_path=corpus_path,
            output_dir=index_dir,
            threads=override_value,
            index_type=cfg.get("index_type", "wiki"),
            retriever_name="bm25"
        )
        t0 = time.time()
        indexer.build_index()
        log_stage("lucene_build", t0, results, index_dir, meta=meta, lucene=True)
    elif retriever == "dpr":
        # DPR Index build
        indexer = DPRIndexer(
            corpus_path=str(corpus_path),
            output_dir=str(index_dir),
            threads=threads,
            batch_size=override_value,
            index_type=cfg.get("index_type", "wiki"),
            retriever_name="dpr"
        )
        t0 = time.time()
        indexer.build_index()
        log_stage("dpr_build", t0, results, index_dir, meta=meta)
    elif retriever == "bge":
        # BGE Index build
        indexer = BGEIndexer(
            corpus_path=str(corpus_path),
            output_dir=str(index_dir),
            index_type="wiki",
            encoder_name="BAAI/bge-large-en-v1.5",
            batch_size=override_value,
            chunk_size=100
        )
        t0 = time.time()
        indexer.build_index()
        log_stage("bge_build", t0, results, index_dir, meta=meta)
    elif retriever == "contriever":
        # Contriever Index build
        idx = ContrieverIndexer(
            corpus_path=str(corpus_path),
            output_dir=str(index_dir),
            index_type="wiki",
            encoder_name="facebook/contriever",
            batch_size=override_value,              # indexing batch (IDs→FAISS), not HF batch
            embedding_batch_size=max(2, min(32, override_value)),  # HF forward batch
        )
        t0 = time.time()
        idx.build_index()
        log_stage("contriever_build", t0, results, index_dir, meta=meta)
    else:
        raise NotImplementedError(f"Retriever '{retriever}' not supported yet.")

    # Save per-run results
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{cfg['name']}_{method}{override_value}_{ts_file}.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(log_path.with_suffix(".csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"✅ Benchmark completed. Logs saved to {log_path}")
    return results


def aggregate_repeats(all_runs):
    """Aggregate repeated runs by stage, computing mean, median and std."""
    summary = []

    # flatten
    flat = [r for run in all_runs for r in run]

    # unique stages
    stages = set(r["stage"] for r in flat)

    for stage in stages:
        stage_rows = [r for r in flat if r["stage"] == stage]

        threads = stage_rows[0]["threads"]
        batch_size = stage_rows[0].get("batch_size", "N/A")

        # Extract lists
        times  = [r["time_s"] for r in stage_rows]
        mems   = [r["memory_mb"] for r in stage_rows]
        sizes  = [r["index_size_mb"] for r in stage_rows if r["index_size_mb"] is not None]

        summary.append({
            "stage": stage,
            "threads": threads,
            "batch_size": batch_size,

            # TIME
            "time_mean_s": round(statistics.mean(times), 2),
            "time_median_s": round(statistics.median(times), 2),
            "time_std_s": round(statistics.stdev(times), 2) if len(times) > 1 else 0.0,

            # MEMORY
            "memory_mean_mb": round(statistics.mean(mems), 2),
            "memory_median_mb": round(statistics.median(mems), 2),
            "memory_std_mb": round(statistics.stdev(mems), 2) if len(mems) > 1 else 0.0,

            # INDEX SIZE
            "index_size_mean_mb": round(statistics.mean(sizes), 2) if sizes else None,
            "index_size_median_mb": round(statistics.median(sizes), 2) if sizes else None,
            "index_size_std_mb": round(statistics.stdev(sizes), 2) if len(sizes) > 1 else 0.0
        })

    return summary



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the benchmark configuration JSON file.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat each benchmark.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
        validate_config(cfg)

    ts = "20251117_215029"#time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_dir = Path(cfg["benchmark"]["log_file"]).with_suffix("") / ts

    # BM 25 in pyserini
    # CONTRIEVER IN format_converters to tsv
    # BGE IN format_converters to tsv

    thread_variants = cfg.get("thread_variants", [8])
    batch_variants = cfg.get("batch_variants", [8])

    print(f"\n📋 Config loaded: {cfg['name']}")
    if cfg.get("retriever", "").lower() == "bm25":
        print(f"→ Thread variants: {thread_variants}")
    else:
        print(f"→ Batch size variants: {batch_variants}")
    print(f"→ Repeats per config: {args.repeats}")
    print(f"→ Results will be saved in: {run_dir}\n")

    method = "threads"
    # ️⃣ Convert corpus once per repeat
    if cfg.get("retriever", "").lower() == "bm25":
        run_lucene_benchmark(cfg, thread_variants)

    # NO THREADS in USAGE -- only batch size
    if cfg.get("retriever", "").lower() == "dpr":
        method = "batch_size"
        run_dpr_benchmark(cfg, batch_variants)

    if cfg.get("retriever", "").lower() == "bge":
        method = "batch_size"
        #run_bge_benchmark(cfg, batch_variants)

    if cfg.get("retriever", "").lower() == "contriever":
        method = "batch_size"
        print("not implemented yet")

    # Merge all summary CSVs
    summary_csvs = [f for f in run_dir.glob("*_summary.csv")]

    if summary_csvs:
        dfs = [pd.read_csv(f) for f in summary_csvs]
        master = pd.concat(dfs, ignore_index=True)
        master_path = run_dir / "master_benchmarks.csv"
        master.to_csv(master_path, index=False)

        print(f"\n📈 Master Summary CSV saved at {master_path}")

        print("\n📊 Generating plots...")
        generate_plots(master_path, save_dir=run_dir / "plots", median=True, group_by=method)
        print("✅ Plots generated.\n")
        print(f"🎉 Benchmarking complete! Results and plots are available in {run_dir.resolve()}")
    else:
        print("⚠️ No summary CSV results found to aggregate.")




