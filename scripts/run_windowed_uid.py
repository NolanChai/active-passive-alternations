"""
Convenience wrapper for windowed UID analysis.

Same flags as run_uid_pipeline.py but with windowed-uid-friendly defaults:
  --uid_level  → "(-10,+10)"
  --uid_unit   → "token"
  --output_dir → "outputs"
  --output_name → "window_uid.csv"

Example (A100):
  uv run python scripts/run_windowed_uid.py data/ gpt2 \
      --context document --uid_unit token --uid_level "(-10,+10)" \
      --generate_counterfactual --fast --batch_size 128 \
      --output_dir outputs --output_name window_uid_tok10.csv \
      --limit_docs 50 --limit_sents_per_doc 12
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(
        description="Run windowed UID analysis (convenience wrapper)."
    )
    # Required
    parser.add_argument("data_dir", type=str,
                        help="Path to folder containing .conllu files.")
    parser.add_argument("model", type=str,
                        help="HuggingFace model name (e.g. gpt2, distilgpt2).")
    # Optional — same as run_uid_pipeline.py, with windowed defaults
    parser.add_argument("--context", "-c", type=str, default=None,
                        help="Context level. Default: all. Common: document.")
    parser.add_argument("--generate_counterfactual", "-cf", action="store_true",
                        help="Generate counterfactual (active↔passive) documents.")
    parser.add_argument("--limit_docs", type=int, default=None,
                        help="Max documents to process.")
    parser.add_argument("--limit_sents_per_doc", type=int, default=None,
                        help="Max sentences per document.")
    parser.add_argument("--uid_level", type=str, default="(-10,+10)",
                        help="Token window for UID analysis (default: (-10,+10)).")
    parser.add_argument("--uid_unit", type=str, default="token",
                        help="Unit for surprisal aggregation: token|word|sentence (default: token).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: cuda|mps|cpu.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory (default: outputs).")
    parser.add_argument("--output_name", type=str, default="window_uid.csv",
                        help="Output CSV filename (default: window_uid.csv).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast bf16/TF32 backend (recommended on A100).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for fast backend (default: 32; use 128 on A100).")
    parser.add_argument("--score_all_cf_sents", action="store_true",
                        help="Score every sentence in CF documents for full trajectory analysis.")

    args, unk = parser.parse_known_args()
    extra_args = {}
    for arg in unk:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if value.lower() == "true":
                extra_args[key] = True
            elif value.lower() == "false":
                extra_args[key] = False
            elif value.isdigit():
                extra_args[key] = int(value)
            elif re.match(r"^-?\d+\.\d+$", value):
                extra_args[key] = float(value)
            else:
                extra_args[key] = value

    from src.uid import run_uid_pipeline

    UD_paths = list(Path(args.data_dir).glob("*.conllu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_name).with_suffix(".csv")
    output_filepath = output_dir / output_file

    uid_dfs = []
    for UD_path in UD_paths:
        uid_df = run_uid_pipeline(
            UD_path,
            model_name=args.model,
            limit_docs=args.limit_docs,
            limit_sents_per_doc=args.limit_sents_per_doc,
            context_levels=args.context,
            generate_counterfactual=args.generate_counterfactual,
            uid_level=args.uid_level,
            uid_unit=args.uid_unit,
            device=args.device,
            output_dir=output_dir,
            output_file=output_file,
            verbose=args.verbose,
            fast=args.fast,
            batch_size=args.batch_size,
            score_all_cf_sents=args.score_all_cf_sents,
        )
        uid_dfs.append(uid_df)

    result = pd.concat(uid_dfs, ignore_index=True)
    result.to_csv(output_filepath)
    print(f"Saved {len(result)} rows → {output_filepath}")


if __name__ == "__main__":
    main()
