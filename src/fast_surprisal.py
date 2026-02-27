"""
Fast surprisal backend for A100/GPU inference.

Drop-in replacement for compute_surprisal() in src/uid.py with:
  - torch.inference_mode() (faster than no_grad)
  - bf16 autocast on CUDA
  - TF32 matmul/cudnn
  - Vectorized surprisal extraction (no Python loop)

Also provides compute_surprisal_batch() for batched inference.
"""

import math
import re
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Enable TF32 on Ampere (A100) and later — no-op on other hardware.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _get_autocast_ctx(device: torch.device):
    """Return bf16 autocast on CUDA, no-op on CPU/MPS."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    # MPS supports float16 but not bfloat16 as of torch 2.x; skip autocast.
    return torch.autocast(device_type="cpu", enabled=False)


def compute_surprisal_fast(
    sentence: str,
    context: str,
    sentences: List[str],
    sent_idx: int,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_len: int = None,
    device: torch.device = None,
    uid_level: str = "sentence",
    verbose: bool = False,
) -> Tuple[List[str], List[float]]:
    """Vectorized, bf16-aware drop-in for compute_surprisal().

    Identical semantics to the original function; replaces the Python surprisal
    loop with a gather operation and wraps the forward pass in inference_mode +
    bf16 autocast on CUDA.

    Args:
        sentence: target sentence text
        context: pre-built context string (output of build_context())
        sentences: all sentences in the current document
        sent_idx: index of `sentence` in `sentences`
        tokenizer: HuggingFace tokenizer
        model: causal LM (GPT-2 family)
        max_len: maximum input length (defaults to tokenizer.model_max_length)
        device: torch device the model lives on
        uid_level: "sentence", "document", or "(-a,+b)" window string
        verbose: print overflow warnings

    Returns:
        (sent_tokens, surprisals) — same as compute_surprisal()
    """
    # --- Import get_input_start_end from the existing module to avoid drift ---
    from src.uid import get_input_start_end

    # Tokenize
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)

    if uid_level not in ["sentence"]:
        document_ids = [
            tokenizer.encode(s, add_special_tokens=False) for s in sentences
        ]
    else:
        document_ids = None

    context_ids = [tokenizer.bos_token_id] + context_ids

    if device is None:
        device = model.device

    input_ids, start, end = get_input_start_end(
        sent_ids, context_ids, document_ids, uid_level, sent_idx
    )

    # Truncate overflow
    max_len = max_len or getattr(tokenizer, "model_max_length", 1024)
    total_len = len(input_ids[0])
    if total_len > max_len:
        if verbose:
            print(f"WARNING: Input length {total_len} exceeds {max_len}. Truncating.")
        overflow = total_len - max_len
        if overflow < len(context_ids):
            input_ids = [input_ids[0][overflow:]]
            start = max(start - overflow, 0)
            end = end - overflow
        else:
            input_ids = [input_ids[0][:-overflow]]
            end = min(end, len(input_ids[0]))

    assert len(input_ids[0]) <= max_len

    input_tensor = torch.tensor(input_ids, device=device)

    with torch.inference_mode(), _get_autocast_ctx(device):
        logits = model(input_tensor).logits  # [1, seq_len, vocab]

    # Vectorized surprisal extraction — equivalent to the original Python loop:
    #   for i in range(start, end): lp = log_probs[0, i-1, input_ids[0][i]]
    window_len = end - start
    if window_len <= 0:
        return [], []

    window_ids = input_tensor[0, start:end]          # [W]
    logits_window = logits[0, start - 1 : end - 1, :]  # [W, V]
    log_probs = torch.log_softmax(logits_window.float(), dim=-1)  # upcast for accuracy
    per_tok_lp = log_probs[torch.arange(window_len, device=device), window_ids]  # [W]
    surprisals = (-per_tok_lp / math.log(2)).tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_tensor[0])
    sent_tokens = tokens[start:end]

    return sent_tokens, surprisals


# ---------------------------------------------------------------------------
# Batched variant — processes multiple (sentence, context) pairs in one pass.
# ---------------------------------------------------------------------------

def _build_one(
    sentence: str,
    context: str,
    sentences: List[str],
    sent_idx: int,
    tokenizer: AutoTokenizer,
    uid_level: str,
    max_len: int,
    verbose: bool,
):
    """Tokenize and build (input_ids_list, start, end) for one item.

    Returns None on overflow/error.
    """
    from src.uid import get_input_start_end

    context_ids = tokenizer.encode(context, add_special_tokens=False)
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)

    if uid_level not in ["sentence"]:
        document_ids = [
            tokenizer.encode(s, add_special_tokens=False) for s in sentences
        ]
    else:
        document_ids = None

    context_ids = [tokenizer.bos_token_id] + context_ids

    try:
        input_ids, start, end = get_input_start_end(
            sent_ids, context_ids, document_ids, uid_level, sent_idx
        )
    except (ValueError, AssertionError) as e:
        if verbose:
            print(f"Skipping item (get_input_start_end error): {e}")
        return None

    total_len = len(input_ids[0])
    if total_len > max_len:
        overflow = total_len - max_len
        if overflow < len(context_ids):
            input_ids = [input_ids[0][overflow:]]
            start = max(start - overflow, 0)
            end = end - overflow
        else:
            input_ids = [input_ids[0][:-overflow]]
            end = min(end, len(input_ids[0]))

    return input_ids[0], start, end


def compute_surprisal_batch(
    items: List[Tuple[str, str, List[str], int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_len: int = None,
    device: torch.device = None,
    uid_level: str = "sentence",
    batch_size: int = 32,
    verbose: bool = False,
) -> List[Tuple[List[str], List[float]]]:
    """Batched surprisal computation for multiple (sentence, context) pairs.

    Pads sequences in each mini-batch with left-padding so that positions are
    right-aligned (GPT-2 is causal; left padding is safe).

    Args:
        items: list of (sentence, context, sentences, sent_idx) tuples
        tokenizer: HuggingFace tokenizer
        model: causal LM
        max_len: max sequence length per item
        device: torch device
        uid_level: uid_level string
        batch_size: number of items per GPU batch
        verbose: print warnings

    Returns:
        List of (sent_tokens, surprisals) in the same order as `items`.
        Items that fail tokenization/extraction return ([], []).
    """
    if device is None:
        device = model.device
    max_len = max_len or getattr(tokenizer, "model_max_length", 1024)

    # Step 1: pre-build all input_ids + (start, end)
    built = []
    for sentence, context, sentences, sent_idx in items:
        result = _build_one(
            sentence, context, sentences, sent_idx,
            tokenizer, uid_level, max_len, verbose
        )
        built.append(result)  # None for failures

    results: List[Tuple[List[str], List[float]]] = [None] * len(built)

    # Step 2: process in mini-batches
    valid_indices = [i for i, b in enumerate(built) if b is not None]

    for batch_start in range(0, len(valid_indices), batch_size):
        batch_indices = valid_indices[batch_start : batch_start + batch_size]
        batch_items = [built[i] for i in batch_indices]

        # Left-pad to the maximum length in this batch
        batch_max_len = max(len(ids) for ids, _, _ in batch_items)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

        padded_ids = []
        pad_offsets = []
        for ids, start, end in batch_items:
            pad_len = batch_max_len - len(ids)
            padded = [pad_id] * pad_len + ids
            padded_ids.append(padded)
            pad_offsets.append(pad_len)

        input_tensor = torch.tensor(padded_ids, device=device)  # [B, T]

        with torch.inference_mode(), _get_autocast_ctx(device):
            logits = model(input_tensor).logits  # [B, T, V]

        for j, (orig_idx, (ids, start, end), pad_off) in enumerate(
            zip(batch_indices, batch_items, pad_offsets)
        ):
            adj_start = start + pad_off
            adj_end = end + pad_off
            window_len = adj_end - adj_start
            if window_len <= 0:
                results[orig_idx] = ([], [])
                continue

            window_ids = input_tensor[j, adj_start:adj_end]
            logits_window = logits[j, adj_start - 1 : adj_end - 1, :]
            log_probs = torch.log_softmax(logits_window.float(), dim=-1)
            per_tok_lp = log_probs[
                torch.arange(window_len, device=device), window_ids
            ]
            surprisals = (-per_tok_lp / math.log(2)).tolist()

            all_tokens = tokenizer.convert_ids_to_tokens(input_tensor[j])
            sent_tokens = all_tokens[adj_start:adj_end]
            results[orig_idx] = (sent_tokens, surprisals)

    # Fill in None failures
    for i, r in enumerate(results):
        if r is None:
            results[i] = ([], [])

    return results
