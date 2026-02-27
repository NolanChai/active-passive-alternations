# Colab / A100 Setup Guide

This guide shows how to run windowed UID analysis on Google Colab (A100 GPU).

---

## 1. Prerequisites

- Google Colab with an A100 runtime (Runtime → Change runtime type → A100)
- Your `.conllu` data files accessible (Google Drive or upload)
- The `colab-monorepo` branch pushed to GitHub

---

## 2. Clone + install

```python
# In a Colab code cell:
import os
!pip install uv -q
if os.path.exists('/content/repo'):
    # Session already has the repo — pull latest changes instead of re-cloning
    !git -C /content/repo pull origin colab-monorepo
else:
    !git clone https://github.com/NolanChai/active-passive-alternations.git /content/repo
    !git -C /content/repo checkout colab-monorepo
%cd /content/repo
!uv sync   # uv downloads Python 3.13 if needed (~1 min first time)
```

---

## 3. Verify CUDA + bf16 support

```python
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("CUDA:", torch.version.cuda)
print("bf16:", torch.cuda.is_bf16_supported())
```

Expected on A100: `bf16: True`. If False (T4), the fast backend falls back to fp32 autocast automatically.

---

## 4. Mount data

**Option A — Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/your_data_folder"   # adjust path
```

**Option B — Upload files:**
```python
import os
from google.colab import files
os.makedirs("/content/active-passive-alternations/data", exist_ok=True)
uploaded = files.upload()   # select .conllu files
# They land in the current directory; move them:
import shutil
for fn in uploaded:
    shutil.move(fn, f"/content/active-passive-alternations/data/{fn}")
DATA_DIR = "/content/active-passive-alternations/data"
```

---

## 5. Run windowed UID

```bash
cd /content/active-passive-alternations
uv run python scripts/run_windowed_uid.py "$DATA_DIR" gpt2 \
    --context document \
    --uid_unit token \
    --uid_level "(-10,+10)" \
    --generate_counterfactual \
    --fast --batch_size 128 \
    --output_dir outputs \
    --output_name window_uid_tok10.csv \
    --limit_docs 50 \
    --limit_sents_per_doc 12
```

---

## 6. Check coverage

```bash
uv run python scripts/check_window_coverage.py \
    --csv outputs/window_uid_tok10.csv
```

---

## 7. Next-sentence analysis (discourse planning)

```bash
uv run python scripts/run_next_sentence_uid.py "$DATA_DIR" gpt2 \
    --context document \
    --uid_unit word \
    --uid_level sentence \
    --sent_offset 1 \
    --generate_counterfactual \
    --fast --batch_size 128 \
    --output_dir outputs \
    --output_name next_sent_uid.csv \
    --limit_docs 50 --limit_sents_per_doc 12
```

Rows contain both factual (`f::...`) and CF (`cf::...`) surprisals for s_{t+1}.
Compare `surp_mean` across factual vs CF rows with the same `sent_idx` to see
whether the active/passive form of s_t affects how surprising s_{t+1} is.
`source_sent_idx` in CF rows records which sentence was converted.

---

## 8. Sweep windows (optional)

```bash
uv run python scripts/sweep_uid_windows.py "$DATA_DIR" gpt2 \
    --windows "(-0,+0)" "(-5,+5)" "(-10,+10)" "(-20,+20)" \
    --context document \
    --generate_counterfactual \
    --fast --batch_size 128 \
    --output_dir outputs \
    --output_name sweep_results.csv \
    --limit_docs 50 --limit_sents_per_doc 12
```

---

## 8. Download results

```python
from google.colab import files
files.download('/content/active-passive-alternations/outputs/window_uid_tok10.csv')
```

---

## Performance notes

| GPU | batch_size | dtype | Est. throughput (GPT-2, seq 512) |
|-----|-----------|-------|----------------------------------|
| A100 | 128 | bf16 | ~400–600 fwd/sec |
| A100 | 32  | fp32 | ~150–250 fwd/sec |
| T4   | 32  | fp16 | ~80–120 fwd/sec  |
| CPU  | 1   | fp32 | ~5–15 fwd/sec    |

**Key bottleneck:** For `--uid_unit token`, the bottleneck is GPU inference. For `--uid_unit word` or `--uid_unit sentence`, stanza NLP (CPU) becomes the bottleneck.

**Recommended settings on A100:**
- `--fast --batch_size 128` for GPT-2
- `--uid_unit token` for fastest runs
- `--context document` gives the most context for windowed UID

**Stanza note:** First run downloads the stanza `en` model (~300 MB). This is cached in `~/stanza_resources/` across sessions if using a persistent disk.
