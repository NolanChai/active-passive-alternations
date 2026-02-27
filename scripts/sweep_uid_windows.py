"""
Sweep multiple uid_level windows and combine results into one CSV.

Example:
  uv run python scripts/sweep_uid_windows.py data/ gpt2 \
      --windows "(-0,+0)" "(-5,+5)" "(-10,+10)" "(-20,+20)" \
      --fast --batch_size 128 \
      --output_dir outputs --output_name sweep_results.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep uid_level windows and collect combined CSV."
    )
    # Required
    parser.add_argument("data_dir", type=str,
                        help="Path to folder containing .conllu files.")
    parser.add_argument("model", type=str,
                        help="HuggingFace model name (e.g. gpt2, distilgpt2).")
    # Window sweep
    parser.add_argument("--windows", nargs="+",
                        default=["(-0,+0)", "(-5,+5)", "(-10,+10)", "(-20,+20)"],
                        help='Windows to sweep (default: "(-0,+0)" "(-5,+5)" "(-10,+10)" "(-20,+20)").')
    # Shared optional args
    parser.add_argument("--context", "-c", type=str, default=None,
                        help="Context level (default: all).")
    parser.add_argument("--generate_counterfactual", "-cf", action="store_true")
    parser.add_argument("--limit_docs", type=int, default=None)
    parser.add_argument("--limit_sents_per_doc", type=int, default=None)
    parser.add_argument("--uid_unit", type=str, default="token",
                        help="token|word|sentence (default: token).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--output_name", type=str, default="sweep_results.csv")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast bf16/TF32 backend.")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    from src.uid import run_uid_pipeline

    UD_paths = list(Path(args.data_dir).iterdir())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / Path(args.output_name).with_suffix(".csv")

    all_rows = []

    for window in args.windows:
        print(f"\n{'='*60}")
        print(f"Window: {window}")
        print(f"{'='*60}")
        window_rows = []
        for UD_path in UD_paths:
            df = run_uid_pipeline(
                UD_path,
                model_name=args.model,
                limit_docs=args.limit_docs,
                limit_sents_per_doc=args.limit_sents_per_doc,
                context_levels=args.context,
                generate_counterfactual=args.generate_counterfactual,
                uid_level=window,
                uid_unit=args.uid_unit,
                device=args.device,
                output_dir=output_dir,
                output_file=Path(f"_sweep_{window.replace(',','_')}.csv"),
                verbose=args.verbose,
                fast=args.fast,
                batch_size=args.batch_size,
            )
            window_rows.append(df)

        window_df = pd.concat(window_rows, ignore_index=True)
        # uid_level is already encoded in each row (from context cfg name), but
        # add an explicit column for easy filtering
        window_df["sweep_window"] = window
        all_rows.append(window_df)

        n = len(window_df)
        mean_surp = window_df["surp_mean"].mean() if "surp_mean" in window_df.columns else float("nan")
        print(f"  → {n} rows, mean surp_mean = {mean_surp:.3f}")

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(output_filepath)
    print(f"\nSaved {len(combined)} total rows → {output_filepath}")

    # Per-window summary table
    if "sweep_window" in combined.columns and "surp_mean" in combined.columns:
        summary = (combined.groupby("sweep_window")["surp_mean"]
                   .agg(["count", "mean", "std"])
                   .rename(columns={"count": "rows", "mean": "surp_mean", "std": "surp_std"}))
        print("\nSummary by window:")
        print(summary.to_string())


if __name__ == "__main__":
    main()
