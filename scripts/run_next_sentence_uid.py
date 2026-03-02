"""
Next-sentence UID analysis (discourse planning experiment).

Tests whether the active/passive form of sentence s_t changes the surprisal
of the following sentence s_{t+1}.  For each CF conversion:

  Factual: P(s_{t+1} | s_0 ... s_t)          — original s_t in context
  CF:      P(s_{t+1} | s_0 ... s_{t-1}, cf(s_t)) — converted s_t in context

The delta reveals whether alternation choice affects how surprising the next
sentence is — a discourse-planning / UID smoothing signal.

The CF documents produced by iter_counterfactual_docs already contain the
full document with one sentence converted, so s_{t+1} is already present.
--sent_offset controls how many sentences ahead to score (default: 1).

Example (A100):
  uv run python scripts/run_next_sentence_uid.py data/ \\
      --context document \\
      --uid_unit word \\
      --uid_level sentence \\
      --sent_offset 1 \\
      --fast --batch_size 128 \\
      --output_dir outputs \\
      --output_name next_sent_uid.csv \\
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
        description="Next-sentence UID analysis: measure s_{t+N} surprisal after CF conversion of s_t."
    )
    # Required
    parser.add_argument("data_dir", type=str,
                        help="Path to folder containing .conllu files.")
    parser.add_argument("model", type=str, nargs="?", default="distilgpt2",
                        help="HuggingFace model name (default: distilgpt2).")
    # Optional
    parser.add_argument("--context", "-c", type=str, default="document",
                        help="Context level (default: document). Must include s_t for next-sentence analysis.")
    parser.add_argument("--limit_docs", type=int, default=None,
                        help="Max documents to process.")
    parser.add_argument("--limit_sents_per_doc", type=int, default=None,
                        help="Max sentences per document.")
    parser.add_argument("--uid_level", type=str, default="sentence",
                        help="UID extraction window (default: sentence). "
                             "Use 'sentence' for pure next-sentence analysis — (-a,+b) windows "
                             "extend into cf(s_t) tokens which confounds the comparison.")
    parser.add_argument("--uid_unit", type=str, default="word",
                        help="Unit for surprisal aggregation: token|word|sentence (default: word).")
    parser.add_argument("--sent_offset", type=int, default=1,
                        help="Sentences ahead of the converted sentence to score (default: 1 = s_{t+1}).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: cuda|mps|cpu.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory (default: outputs).")
    parser.add_argument("--output_name", type=str, default="next_sent_uid.csv",
                        help="Output CSV filename (default: next_sent_uid.csv).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Use fast bf16/TF32 backend (recommended on A100).")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for fast backend (default: 32; use 128 on A100).")

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

    if args.sent_offset < 1:
        parser.error("--sent_offset must be >= 1 for next-sentence analysis. "
                     "Use run_windowed_uid.py for sent_offset=0.")

    from src.uid import run_uid_pipeline

    UD_paths = list(Path(args.data_dir).glob("*.conllu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_name).with_suffix(".csv")
    output_filepath = output_dir / output_file

    print(f"Next-sentence UID analysis (sent_offset={args.sent_offset})")
    print(f"  Scoring s_{{t+{args.sent_offset}}} after CF conversion of s_t")
    print(f"  uid_unit={args.uid_unit}  uid_level={args.uid_level}  context={args.context}")
    print()

    uid_dfs = []
    for UD_path in UD_paths:
        uid_df = run_uid_pipeline(
            UD_path,
            model_name=args.model,
            limit_docs=args.limit_docs,
            limit_sents_per_doc=args.limit_sents_per_doc,
            context_levels=args.context,
            generate_counterfactual=True,   # always required for next-sentence
            uid_level=args.uid_level,
            uid_unit=args.uid_unit,
            device=args.device,
            output_dir=output_dir,
            output_file=output_file,
            verbose=args.verbose,
            fast=args.fast,
            batch_size=args.batch_size,
            sent_offset=args.sent_offset,
        )
        uid_dfs.append(uid_df)

    result = pd.concat(uid_dfs, ignore_index=True)
    result.to_csv(output_filepath)
    print(f"Saved {len(result)} rows → {output_filepath}")


if __name__ == "__main__":
    main()
