import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from conllu import parse_incr
from transformers import AutoModelForCausalLM, AutoTokenizer

# moved from UID.ipynb -- will clean up soon

# UID Pipeline
# This is a notebook to build and compute UID metrics at multiple context levels:
# - Sentence-only (no prior context)
# - Previous sentence(s) (local discourse)
# - Document-level (full prior discourse)


def sent_text(sent):
    toks = [t for t in sent if isinstance(t.get("id"), int)]
    return " ".join(t["form"] for t in toks)


def iter_ud_docs(path, limit_docs=None, limit_sents_per_doc=None):
    docs = []
    current_id = None
    current = []

    with open(path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            meta = sent.metadata
            if "newdoc id" in meta:
                if current_id is not None and current:
                    docs.append((current_id, current))
                    if limit_docs and len(docs) >= limit_docs:
                        return docs
                current_id = meta["newdoc id"]
                current = []

            text = meta.get("text") or sent_text(sent)
            if text:
                current.append(text)
                if limit_sents_per_doc and len(current) >= limit_sents_per_doc:
                    docs.append((current_id or "doc", current))
                    if limit_docs and len(docs) >= limit_docs:
                        return docs
                    current = []

        if current:
            docs.append((current_id or "doc", current))

    return docs


def load_lm(model_name="distilgpt2"):
    # note - using distilgpt for fast prototyping, use gpt-2 for final
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    model.eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # metal for mac
    else:
        device = torch.device("cpu")

    model.to(device)
    return tokenizer, model, device


def build_context(
    sents,
    idx,
    mode,
    k=None,
    max_tokens=None,
    window=None,
    tokenizer=None,
):
    # mode: none, prev, doc, window
    if mode == "none" or idx == 0:
        return ""

    if mode == "prev":
        k = k or 1
        start = max(0, idx - k)
        return " ".join(sents[start:idx])

    if mode == "doc":
        return " ".join(sents[:idx])

    if mode == "window":
        if window is None:
            return ""
        wtype = window.get("type", "sent")
        side = window.get("side", "left")
        left = window.get("left", 0)
        right = window.get("right", 0)
        size = window.get("size")
        rng = window.get("range")

        if rng is not None:
            if isinstance(rng, (list, tuple)) and len(rng) == 2:
                left, right = rng
            else:
                raise ValueError("window['range'] must be a 2-tuple like (left, right)")

        if size is not None:
            if side == "left":
                left = size
                right = 0
            elif side == "right":
                left = 0
                right = size
            else:
                left = size
                right = size

        if wtype == "sent":
            start = max(0, idx - left)
            end = min(len(sents), idx + right + 1)
            parts = sents[start:end]
            # remove current sentence from context
            if 0 <= idx - start < len(parts):
                parts = parts[: idx - start] + parts[idx - start + 1 :]
            return " ".join(parts)

        if wtype == "token":
            if tokenizer is None:
                raise ValueError("tokenizer is required for token window context")
            left_sents = " ".join(sents[:idx])
            right_sents = " ".join(sents[idx + 1 :])
            left_ids = tokenizer.encode(left_sents, add_special_tokens=False)
            right_ids = tokenizer.encode(right_sents, add_special_tokens=False)
            left_ids = left_ids[-left:] if left > 0 else []
            right_ids = right_ids[:right] if right > 0 else []
            return tokenizer.decode(left_ids + right_ids)

    return ""


# token level surprisal for the current sentence only
def compute_surprisal(sentence, context, tokenizer, model, max_len=None, device=None):
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)

    # Add BOS to allow a probability for the first token
    context_ids = [tokenizer.bos_token_id] + context_ids

    max_len = max_len or getattr(tokenizer, "model_max_length", 1024)
    total_len = len(context_ids) + len(sent_ids)
    if total_len > max_len:
        overflow = total_len - max_len
        if overflow < len(context_ids):
            context_ids = context_ids[overflow:]
        else:
            sent_ids = sent_ids[-max_len + 1:]
            context_ids = [tokenizer.bos_token_id]

    if device is None:
        device = model.device

    input_ids = torch.tensor([context_ids + sent_ids], device=device)

    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)

    start = len(context_ids)
    surprisals = []
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    for i in range(start, input_ids.size(1)):
        token_id = input_ids[0, i]
        lp = log_probs[0, i - 1, token_id]
        surprisal = (-lp / math.log(2)).item()
        surprisals.append(surprisal)

    sent_tokens = tokens[start:]
    return sent_tokens, surprisals


# Really basic UID metrics
def uid_metrics(surprisals):
    if not surprisals:
        return {}
    arr = np.array(surprisals)
    mean = arr.mean()
    std = arr.std()
    mad = np.mean(np.abs(arr - mean))
    rng = arr.max() - arr.min()

    x = np.arange(len(arr))
    slope = np.polyfit(x, arr, 1)[0] if len(arr) > 1 else 0.0

    return {
        "uid_mean": mean,
        "uid_std": std,
        "uid_mad": mad,
        "uid_range": rng,
        "uid_cv": std / mean if mean > 0 else 0.0,
        "uid_slope": slope,
        "uid_len": len(arr),
    }


def run_uid_pipeline(
    ud_path,
    model_name="distilgpt2",
    limit_docs=3,
    limit_sents_per_doc=8,
    context_levels=None,
):
    docs = iter_ud_docs(ud_path, limit_docs=limit_docs, limit_sents_per_doc=limit_sents_per_doc)
    tokenizer, model, device = load_lm(model_name=model_name)

    if context_levels is None:
        context_levels = [
            {"name": "sentence", "mode": "none"},
            {"name": "prev1", "mode": "prev", "k": 1},
            {"name": "prev3", "mode": "prev", "k": 3},
            {"name": "document", "mode": "doc"},
            
            # sent[-L,+R] = sentence window with L sentences before, R after
            # tok[-L,+R]  = token window with L tokens before, R after
            {"name": "sent[-2,+0]", "mode": "window", "window": {"type": "sent", "side": "left", "size": 2}},
            {"name": "sent[-2,+2]", "mode": "window", "window": {"type": "sent", "side": "both", "size": 2}},
            {"name": "tok[-64,+0]", "mode": "window", "window": {"type": "token", "side": "left", "size": 64}},
            {"name": "tok[-64,+64]", "mode": "window", "window": {"type": "token", "side": "both", "size": 64}},
        ]

    rows = []
    for doc_id, sents in docs:
        for i, sent in enumerate(sents):
            for cfg in context_levels:
                context = build_context(
                    sents,
                    i,
                    mode=cfg["mode"],
                    k=cfg.get("k"),
                    max_tokens=cfg.get("max_tokens"),
                    window=cfg.get("window"),
                    tokenizer=tokenizer,
                )
                tokens, surprisals = compute_surprisal(
                    sent, context, tokenizer, model, device=device
                )
                metrics = uid_metrics(surprisals)
                row = {
                    "doc_id": doc_id,
                    "sent_idx": i,
                    "context": cfg["name"],
                    "sentence": sent,
                }
                row.update(metrics)
                rows.append(row)

    uid_df = pd.DataFrame(rows)
    return uid_df
