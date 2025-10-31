import pandas as pd
from pathlib import Path
import argparse

def collect_logs(log_dir, output_file="master_benchmarks.csv"):
    log_dir = Path(log_dir)
    all_files = list(log_dir.glob("*.csv"))

    if not all_files:
        raise FileNotFoundError(f"No CSV logs found in {log_dir}")

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            df["source_file"] = f.name  # optional: Herkunft des Logs
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Could not read {f}: {e}")

    master = pd.concat(dfs, ignore_index=True)

    out_path = Path(output_file)
    master.to_csv(out_path, index=False)

    print(f"✅ Collected {len(all_files)} log files into {out_path}")
    return master


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True, help="Directory with benchmark CSV logs")
    parser.add_argument("--output_file", default="master_benchmarks.csv", help="Path for combined output CSV")
    args = parser.parse_args()

    collect_logs(args.log_dir, args.output_file)
