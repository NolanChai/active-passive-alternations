"""
Coverage and sufficiency checker for windowed UID CSV outputs.

Validates:
  1. Row counts by uid_level / context / uid_unit
  2. raw_surps distribution (len of window vectors)
  3. For (-a,+b) uid_levels: confirms context is actually included
  4. Factual↔counterfactual pairing completeness

Usage:
  uv run python scripts/check_window_coverage.py --csv outputs/window_uid_tok10.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import ast
import json
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_raw_surps(val):
    """Robustly parse a raw_surps cell (JSON list or Python repr)."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str):
        return None
    val = val.strip()
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        pass
    try:
        return ast.literal_eval(val)
    except Exception:
        return None


def parse_doc_id(doc_id: str):
    """Parse doc_id into (fact, doc_name, conv_id, conv_type).

    Format: f::docname::-1::og  or  cf::docname::5::p>a
    Returns None for unparseable ids.
    """
    parts = str(doc_id).split("::")
    if len(parts) != 4:
        return None
    fact, doc_name, conv_id, conv_type = parts
    return fact, doc_name, conv_id, conv_type


def window_spec(uid_level: str):
    """Return (before, after) int pair for a windowed uid_level, else None."""
    m = re.search(r"[\(\[]\-(\d+),\s*\+(\d+)[\)\]]", uid_level)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Check windowed UID CSV coverage.")
    parser.add_argument("--csv", required=True, help="Path to output CSV file.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n{'='*60}")
    print(f"  File: {csv_path}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Row counts by group
    # ------------------------------------------------------------------
    group_cols = [c for c in ["uid_level", "context", "uid_unit"] if c in df.columns]
    if group_cols:
        print("Row counts by group:")
        counts = df.groupby(group_cols).size().reset_index(name="rows")
        print(counts.to_string(index=False))
    else:
        print("(No uid_level/context/uid_unit columns found for grouping)")
    print()

    # ------------------------------------------------------------------
    # 2. raw_surps distribution
    # ------------------------------------------------------------------
    if "raw_surps" not in df.columns:
        print("WARNING: 'raw_surps' column not found. Was this generated with a recent pipeline version?")
        print("  → Re-run with current code to include raw_surps.\n")
    else:
        parsed = df["raw_surps"].map(parse_raw_surps)
        n_parseable = parsed.notna().sum()
        lengths = parsed.dropna().map(len)
        print(f"raw_surps: {n_parseable}/{len(df)} rows parseable")
        if len(lengths) > 0:
            print(f"  len distribution: mean={lengths.mean():.1f}  "
                  f"p25={lengths.quantile(0.25):.0f}  "
                  f"p75={lengths.quantile(0.75):.0f}  "
                  f"max={lengths.max():.0f}")
        print()

        # ------------------------------------------------------------------
        # 3. Window context check — for each (-a,+b) uid_level
        # ------------------------------------------------------------------
        uid_levels = df["uid_level"].unique() if "uid_level" in df.columns else []
        for ul in uid_levels:
            spec = window_spec(str(ul))
            if spec is None:
                continue
            before, after = spec
            target_extra = before + after
            if target_extra == 0:
                print(f"  uid_level={ul}: no context tokens requested (0,0) — skipping window check.")
                continue

            subset = df[df["uid_level"] == ul]
            subset_parsed = parsed.loc[subset.index].dropna()

            if len(subset_parsed) == 0:
                print(f"  uid_level={ul}: no parseable raw_surps rows.")
                continue

            # Estimate baseline sentence length (uid_level="sentence" rows, same doc)
            # Fall back to uid_len column if available
            if "uid_len" in df.columns:
                baseline = df[df["uid_level"] == ul]["uid_len"].median()
            else:
                baseline = subset_parsed.map(len).median() * 0.5  # rough heuristic

            lengths_sub = subset_parsed.map(len)
            frac_extended = (lengths_sub > baseline).mean()
            print(f"  uid_level={ul}: {len(subset)} rows, "
                  f"{frac_extended*100:.1f}% have len(raw_surps) > baseline ({baseline:.0f})")
            if frac_extended < 0.5:
                print(f"    WARNING: fewer than 50% of rows show extended context. "
                      f"Is '--context document' being used?")
        print()

    # ------------------------------------------------------------------
    # 4. Factual↔counterfactual pairing completeness
    # ------------------------------------------------------------------
    if "doc_id" not in df.columns:
        print("WARNING: 'doc_id' column not found — cannot compute pairing stats.")
        return

    parsed_ids = df["doc_id"].map(parse_doc_id)
    n_parseable_ids = parsed_ids.notna().sum()
    print(f"doc_id parsing: {n_parseable_ids}/{len(df)} rows parseable")

    if n_parseable_ids == 0:
        print("  Could not parse any doc_ids — skipping pairing analysis.")
        return

    df2 = df.copy()
    df2[["fact", "doc_name", "conv_id", "conv_type"]] = pd.DataFrame(
        parsed_ids.dropna().tolist(), index=parsed_ids.dropna().index
    )

    factual = df2[df2["fact"] == "f"]
    counterfactual = df2[df2["fact"] == "cf"]

    print(f"  Factual rows:        {len(factual)}")
    print(f"  Counterfactual rows: {len(counterfactual)}")

    if len(counterfactual) == 0:
        print("  No counterfactual rows found. Run with --generate_counterfactual.")
        return

    # Build factual key set
    match_cols = ["doc_name", "conv_id"]
    opt_cols = [c for c in ["context", "uid_unit"] if c in df2.columns]
    match_cols += opt_cols

    factual_keys = set(
        zip(*[factual[c] for c in match_cols])
    )

    def has_match(row):
        key = tuple(row[c] for c in match_cols)
        return key in factual_keys

    cf_matched = counterfactual.apply(has_match, axis=1).sum()
    pct = cf_matched / len(counterfactual) * 100
    print(f"  Pairing completeness: {cf_matched}/{len(counterfactual)} cf rows "
          f"have a matching factual row ({pct:.1f}%)")

    if pct < 50:
        print("  WARNING: <50% pairing. Possible causes:")
        print("    - Factual docs were filtered differently from CF docs")
        print("    - doc_id conv_id mismatch (check passivization logic)")

    # Per conv_type breakdown
    if "conv_type" in df2.columns:
        print("\n  Counterfactual rows by conv_type:")
        print(counterfactual["conv_type"].value_counts().to_string())

    print()
    print("Done.")


if __name__ == "__main__":
    main()
