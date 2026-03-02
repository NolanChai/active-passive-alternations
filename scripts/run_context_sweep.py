"""
Context-level sweep for UID analysis.

This script implements the *conditioning context* sweep proposed by the
collaborator: vary how much left context the model is given when scoring s_t,
while ALWAYS scoring only s_t tokens (uid_level="sentence").

This separates two quantities that the windowed sweep conflates:
  - How much the model *knows* before scoring s_t  ← varied here
  - Which tokens are *measured*                    ← always s_t only

Expected result: the CF–factual delta grows as more left context is given,
because the model develops stronger expectations about which voice is
appropriate at this discourse position.

Output CSV columns mirror the other sweep CSVs (surp_mean, uid_std, etc.) with
the `context` column encoding the conditioning amount:
  no_ctx, tok[-10,+0], tok[-20,+0], tok[-50,+0], tok[-100,+0], tok[-200,+0],
  document

Example (A100):
  uv run python scripts/run_context_sweep.py data/ \\
      --generate_counterfactual --fast --batch_size 128 \\
      --output_dir outputs --output_name sweep_context_word.csv

Token-level output:
  uv run python scripts/run_context_sweep.py data/ \\
      --generate_counterfactual --uid_unit token --fast --batch_size 128 \\
      --output_dir outputs --output_name sweep_context_tok.csv

Custom context levels:
  uv run python scripts/run_context_sweep.py data/ \\
      --generate_counterfactual \\
      --context_levels no_ctx sent[-1,+0] sent[-3,+0] sent[-5,+0] document \\
      --output_dir outputs --output_name sweep_context_sent.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import re
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

# Default context levels for the sweep — spans from no conditioning context
# to full preceding document.  Uses token-based sizes for consistent comparison
# across documents with varying sentence lengths.
DEFAULT_CONTEXT_LEVELS = [
    "no_ctx",
    "tok[-10,+0]",
    "tok[-20,+0]",
    "tok[-50,+0]",
    "tok[-100,+0]",
    "tok[-200,+0]",
    "document",
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Context-level sweep: score only s_t tokens across varying amounts "
            "of left conditioning context. Measures how discourse context "
            "amplifies the CF–factual UID difference."
        )
    )
    # Required
    parser.add_argument("data_dir", type=str,
                        help="Path to folder containing .conllu files.")
    parser.add_argument("model", type=str, nargs="?", default="distilgpt2",
                        help="HuggingFace model name (default: distilgpt2).")
    # Optional
    parser.add_argument(
        "--context_levels", nargs="+",
        default=DEFAULT_CONTEXT_LEVELS,
        metavar="LEVEL",
        help=(
            "Context levels to sweep. Space-separated list. "
            "Available: no_ctx, tok[-10,+0], tok[-20,+0], tok[-50,+0], "
            "tok[-100,+0], tok[-200,+0], sent[-1,+0], sent[-3,+0], "
            "sent[-5,+0], sent[-10,+0], document. "
            f"Default: {' '.join(DEFAULT_CONTEXT_LEVELS)}"
        ),
    )
    parser.add_argument("--generate_counterfactual", "-cf", action="store_true",
                        help="Generate counterfactual (active↔passive) documents.")
    parser.add_argument("--limit_docs", type=int, default=None,
                        help="Max documents to process.")
    parser.add_argument("--limit_sents_per_doc", type=int, default=None,
                        help="Max sentences per document.")
    parser.add_argument("--uid_unit", type=str, default="word",
                        help="Unit for surprisal aggregation: token|word|sentence (default: word).")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device: cuda|mps|cpu.")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory (default: outputs).")
    parser.add_argument("--output_name", type=str, default="sweep_context_word.csv",
                        help="Output CSV filename (default: sweep_context_word.csv).")
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

    from src.uid import run_uid_pipeline

    UD_paths = list(Path(args.data_dir).glob("*.conllu"))
    if not UD_paths:
        print(f"ERROR: no .conllu files found in {args.data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_name).with_suffix(".csv")
    output_filepath = output_dir / output_file

    print("Context-level sweep (uid_level=sentence — only s_t tokens scored)")
    print(f"  Context levels : {args.context_levels}")
    print(f"  uid_unit       : {args.uid_unit}")
    print(f"  Documents      : {len(UD_paths)} .conllu files")
    print(f"  Output         : {output_filepath}")
    print()

    uid_dfs = []
    for UD_path in UD_paths:
        uid_df = run_uid_pipeline(
            UD_path,
            model_name=args.model,
            limit_docs=args.limit_docs,
            limit_sents_per_doc=args.limit_sents_per_doc,
            # Pass the full list — run_uid_pipeline scores every context level
            # per sentence in a single pass, so cost = 1 forward pass per
            # (sentence × context_level) regardless of how many levels we sweep.
            context_levels=args.context_levels,
            generate_counterfactual=args.generate_counterfactual,
            # CRITICAL: always "sentence" — we score only s_t tokens.
            # The context_levels list above controls the conditioning context.
            uid_level="sentence",
            uid_unit=args.uid_unit,
            device=args.device,
            output_dir=output_dir,
            output_file=output_file,
            verbose=args.verbose,
            fast=args.fast,
            batch_size=args.batch_size,
        )
        uid_dfs.append(uid_df)

    result = pd.concat(uid_dfs, ignore_index=True)
    result.to_csv(output_filepath)
    print(f"Saved {len(result):,} rows → {output_filepath}")


if __name__ == "__main__":
    main()
