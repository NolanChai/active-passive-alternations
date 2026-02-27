#!/usr/bin/env bash
# Bootstrap script for Google Colab / fresh Linux environment.
# Run from the repo root:  bash colab/bootstrap.sh

set -e

echo "=== Installing uv ==="
pip install uv -q

echo "=== Installing dependencies ==="
uv sync

echo "=== CUDA diagnostics ==="
python - <<'EOF'
import torch
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA avail: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"CUDA:     {torch.version.cuda}")
    print(f"bf16:     {torch.cuda.is_bf16_supported()}")
EOF

echo ""
echo "=== Ready. Example smoke run: ==="
echo ""
echo "  uv run python scripts/run_windowed_uid.py data/ distilgpt2 \\"
echo "      --context document --uid_level '(-10,+10)' \\"
echo "      --generate_counterfactual --fast --batch_size 32 \\"
echo "      --output_dir outputs --output_name smoke_test.csv \\"
echo "      --limit_docs 2 --limit_sents_per_doc 5"
echo ""
echo "  uv run python scripts/check_window_coverage.py --csv outputs/smoke_test.csv"
