"""
Analyze Rankify Benchmark Results
---------------------------------
Supports plotting based on mean, median, or both (--all).
Automatically saves plots in a 'plots/' subfolder inside the benchmark directory,
unless a custom --save-dir is provided.

Smart behavior:
- Skips batch-based plots if batch_size has no variation (e.g. for BM25).
- Removes the batch_size legend from thread plots if irrelevant.

Usage examples:
    python analyze_benchmarks.py --file logs/.../master_benchmarks.csv --median
    python analyze_benchmarks.py --file logs/.../master_benchmarks.csv --mean
    python analyze_benchmarks.py --file logs/.../master_benchmarks.csv --all
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def generate_plots(file, mean=False, median=False, all=False, save_dir=None, group_by="threads"):
    # Arguments are now passed directly to the function
    class Args:
        pass
    args = Args()

    args.file = file
    args.mean = mean
    args.median = median
    args.all = all
    args.save_dir = save_dir
    args.group_by = group_by

    if group_by not in ["threads", "batch_size"]:
        raise ValueError("group_by must be 'threads' or 'batch_size'")

    # --- Resolve paths ---
    csv_path = Path(args.file)
    if not csv_path.exists():
        raise FileNotFoundError(f"❌ Could not find {csv_path}")

    # Default: create plots directory next to CSV
    save_dir = Path(args.save_dir) if args.save_dir else csv_path.parent / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading benchmark data from: {csv_path}")
    print(f"📁 Saving plots to: {save_dir.resolve()}")
    print(f"🔀 Grouping by: {args.group_by}")

    # --- Load data ---
    df = pd.read_csv(csv_path).dropna(subset=["stage"])

    # Choose which values to plot based on mode
    if args.median:
        df["time_plot"] = df["time_median_s"]
        df["memory_plot"] = df["memory_median_mb"]
        df["index_size_plot"] = df["index_size_median_mb"]
        plot_suffix = "Median"
    else:
        df["time_plot"] = df["time_mean_s"]
        df["memory_plot"] = df["memory_mean_mb"]
        df["index_size_plot"] = df["index_size_mean_mb"]
        plot_suffix = "Mean"


    # include only rows related to the Lucene build stage.
    lucene_df = df.copy()

    sns.set(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (8, 5)

    # --- Aggregation (dynamic) ---
    agg = (
        # group by threads or batch_size
        lucene_df.groupby([group_by])
        .agg(
            time_median=("time_plot", "median"),
            time_mean=("time_plot", "mean"),
            time_std=("time_plot", "std"),
            memory_median=("memory_plot", "median"),
            memory_mean=("memory_plot", "mean"),
            memory_std=("memory_plot", "std"),
            index_size_median=("index_size_plot", "median"),
            index_size_mean=("index_size_plot", "mean"),
        )
        .reset_index()
    )

    if "batch_size" not in agg.columns:
        agg["batch_size"] = "N/A"

    agg = agg.fillna(0)

    print("\n📊 Summary:")
    print(agg.round(3).to_string(index=False))

    # === Dynamic column names ===
    x_col = args.group_by
    x_label = "Threads" if args.group_by == "threads" else "Batch Size"

    # === Plot helper ===
    def make_plot(metric, ylabel, value_col, prefix):
        plt.figure()
        sns.lineplot(data=agg, x=x_col, y=value_col, marker="o")
        plt.title(f"{metric} vs. {x_label} ({plot_suffix})")
        plt.xlabel(x_label)
        plt.ylabel(ylabel)
        plt.tight_layout()

        out_path = save_dir / f"{prefix}_vs_{x_col}_{plot_suffix.lower()}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"💾 Saved {out_path}")

        # === Plotting logic ===
    if not any([args.mean, args.median, args.all]):
        args.median = True

    def make_thread_plot(metric, ylabel, title_suffix, value_col, prefix, has_batch_variation=False):
        plt.figure()
        if has_batch_variation:
            sns.lineplot(data=agg, x="threads", y=value_col, hue="batch_size", marker="o")
            plt.legend(title="Batch Size")
        else:
            sns.lineplot(data=agg, x="threads", y=value_col, marker="o")
            plt.legend().remove()
        plt.title(f"{metric} vs. Threads ({title_suffix})")
        plt.xlabel("Threads")
        plt.ylabel(ylabel)
        plt.tight_layout()
        out_path = save_dir / f"{prefix}_vs_threads_{title_suffix.lower()}.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"💾 Saved {out_path}")

    # --- Plotting logic ---

    #Determine plotting modes
    plot_mean = args.mean or args.all
    plot_median = args.median or args.all

    # ---- PLOT GENERATION ----
    if plot_mean:
        print("\n📈 Generating mean-based plots...")
        make_plot("Build Time", "Time [s]", "time_mean", "build_time_mean")
        make_plot("Memory", "Memory [MB]", "memory_mean", "memory_mean")
        make_plot("Index Size", "Index Size [MB]", "index_size_mean", "index_size_mean")

    if plot_median:
        print("\n📈 Generating median-based plots...")
        make_plot("Build Time", "Time [s]", "time_median", "build_time_median")
        make_plot("Memory", "Memory [MB]", "memory_median", "memory_median")
        make_plot("Index Size", "Index Size [MB]", "index_size_median", "index_size_median")

    print("\n✅ Analysis complete.")
    print(f"All plots saved to: {save_dir.resolve()}")

if __name__ == "__main__":
    generate_plots()
