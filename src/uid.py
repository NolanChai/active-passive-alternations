import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
from conllu import parse, parse_incr
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import stanza
stanza.download('en')
nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,mwt,pos,lemma,depparse"
)

from src.units import Document, Sentence, Word
from src.unigram import UnigramLM
from src.utils import *

# moved from UID.ipynb -- will clean up soon

# UID Pipeline
# - Sentence-only (no prior context)
# - Previous sentence(s) (local discourse)
# - Document-level (full prior discourse)
# - Windowed context (e.g. +/- n sentences, +/- n tokens)


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

def extend_doc_list(docs, current, current_id):
    curr_doc = Document(current)
    converted_docs = curr_doc.convert_all()
    curr_doc_text = [s.text for s in curr_doc]
    docs.append((f'f::{current_id}::-1::og', curr_doc_text))
    
    if len(converted_docs) > 0:
        counterf_docs, pass_idxs, conversions = zip(*converted_docs)
        counterf_ids = (f'cf::{current_id}::{pass_idx}::{conv}' 
                        for pass_idx, conv in zip(pass_idxs, conversions))
        counterf_doc_texts = [[s.text for s in counterf_doc] 
                            for counterf_doc in counterf_docs]
        counterf_doc_tuples = zip(counterf_ids, counterf_doc_texts)
        docs.extend(counterf_doc_tuples)

def iter_counterfactual_docs(path, limit_docs=None, limit_sents_per_doc=None):
    docs = []
    current_id = None
    current = []

    with open(path, "r", encoding="utf-8") as f:
        for sent in parse_incr(f):
            meta = sent.metadata
            if "newdoc id" in meta:
                if current_id is not None and current:
                    extend_doc_list(docs, current, current_id)
                    if limit_docs and len(docs) >= limit_docs:
                        return docs
                current_id = meta["newdoc id"]
                current = []

            current.append(sent)
            if limit_sents_per_doc and len(current) >= limit_sents_per_doc:
                current[0].metadata.update({"newdoc id": current_id})
                extend_doc_list(docs, current, current_id)
                if limit_docs and len(docs) >= limit_docs:
                    return docs
                current = []

        if current:
            extend_doc_list(docs, current, current_id)

    return docs
    


def load_lm(model_name="distilgpt2", device=None):
    # note - using distilgpt for fast prototyping, use gpt-2 for final
    assert device is not None, "Please specify device."
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    model.eval()

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

def truncate_doc_ids(document_ids, trunc_from_left=0, trunc_from_right=0):
    result = []
    total_len = sum([len(sent) for sent in document_ids])
    result_len = total_len - trunc_from_left - trunc_from_right
    assert result_len > 0, "Cannot truncate more than document length!"
    
    trunc_left_remaining = trunc_from_left
    curr_len = 0
    for sent in document_ids:
        curr_sent = []
        
        for tok_id in sent:
            if curr_len >= result_len:
                break
            if trunc_left_remaining:
                trunc_left_remaining -= 1
                continue
            curr_sent.append(tok_id)
            curr_len += 1
            
        if curr_sent:
            result.append(curr_sent)
        if curr_len >= result_len:
            break
    return result
        

# token level surprisal for the current sentence only
def compute_surprisal(sentence,
                      context,
                      sentences,
                      sent_idx,
                      tokenizer,
                      model,
                      max_len=None,
                      device=None,
                      uid_level="sentence",
                      verbose=False):
    """Compute the surprisal for the given sentence with a causal LM, given
    context level and uid level.

    Args:
        sentence (str): sentence to compute
        context (str): context level to give the model before the sentence
        sentences (List[str]): list of all sentences in the current document
        sent_idx (int): index of `sentence` within `sentences`
        tokenizer (AutoTokenizer): tokenizer for the model
        model (AutoModelForCausalLM): model to use for surprisal calc
        max_len (int, optional): max length of the context + sentence. Defaults to None.
        device (str, optional): device that the model is on. Defaults to None.
        uid_level (str, optional): level at which to calculate UID. Defaults to "sentence".

    Returns:
        (List[int], List[float]): token ids and surprisals at the given UID level
    """
    # tokenize
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    sent_ids = tokenizer.encode(sentence, add_special_tokens=False)
    
    if uid_level not in ["sentence"]:
        document_ids = [tokenizer.encode(sent, add_special_tokens=False)
                        for sent in sentences]
    else:
        document_ids = None
    # Add BOS to allow a probability for the first token
    context_ids = [tokenizer.bos_token_id] + context_ids

    if device is None:
        device = model.device

    # get proper input for the model based on context level 
    # and the proper extracted window based on uid level
    input_ids, start, end = get_input_start_end(sent_ids, context_ids, document_ids,
                                                uid_level, sent_idx)
    # Deal with overflow + length
    max_len = max_len or getattr(tokenizer, "model_max_length", 1024)
    total_len = len(input_ids[0])
    if total_len > max_len:
        if verbose:
            print(f"WARNING: Input length {total_len} exceeds model context length {max_len}.")
            print("Resizing inputs to match...")
        overflow = total_len - max_len
        if overflow < len(context_ids):
            input_ids = [input_ids[0][overflow:]]
            start = max(start - overflow, 0)
            end = end - overflow
        else:
            input_ids = [input_ids[0][:-overflow]]
            end = min(end, len(input_ids[0]))
        if verbose:
            print(f"New input length: {len(input_ids[0])}")

    assert len(input_ids[0]) <= max_len, "Input Length Mismatch"
    # run through model
    input_ids = torch.tensor(input_ids, device=device)
    with torch.no_grad():
        logits = model(input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)
    
    surprisals = []

    # compute surprisals
    for i in range(start, end):
        try:
            token_id = input_ids[0, i]
            lp = log_probs[0, i - 1, token_id]
            surprisal = (-lp / math.log(2)).item()
            surprisals.append(surprisal)
        except IndexError:
            continue

    # re-code tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    sent_tokens = tokens[start:end]
    
    return sent_tokens, surprisals

def get_input_start_end(sent_ids,
                        context_ids,
                        doc_ids,
                        uid_level,
                        sent_idx):
    if uid_level == "sentence":
        input_ids = [context_ids + sent_ids]
        start = len(context_ids)
        end = len(input_ids[0])
    elif uid_level == "document":
        input_ids = [[id for sent in doc_ids for id in sent]]
        start = 0
        end = len(input_ids[0])
    elif key := re.search(r"[\(\[]\-(\d+)\, *\+(\d+)[\)\]]", uid_level):
        before, after = key.groups()
        before = int(before)
        after = int(after)
        if before > len(context_ids):
            raise ValueError("Left uid calculation window cannot extend past context.")
        right = [id for sent in doc_ids[sent_idx+1:] for id in sent]
        input_ids = [context_ids + sent_ids + right[:after]]
        start = len(context_ids) - before
        end = len(input_ids[0])
    else:
        raise ValueError(f"UID Level {uid_level} not supported.")
        
    return input_ids, start, end
    
# Really basic UID metrics
def uid_metrics(surprisals, 
                uni_probs):
    if not surprisals:
        return {}
    arr = np.array(surprisals)
    uni_arr = np.array(uni_probs)
    uni_arr = -np.log(uni_arr + 1e-5) / np.log(2) # probs -> surps
    mean = arr.mean()
    slor = (uni_arr.sum() - arr.sum()) / len(arr)
    std = arr.std()
    mad = np.mean(np.abs(arr - mean))
    pwd = (np.diff(arr) ** 2).mean()
    rng = arr.max() - arr.min()

    x = np.arange(len(arr))
    slope = np.polyfit(x, arr, 1)[0] if len(arr) > 1 else 0.0

    return {
        "raw_surps": list(surprisals),
        "surp_mean": mean,
        "surp_slor": slor,
        "uid_std": std,
        "uid_mad": mad,
        "uid_pwd": pwd,
        "uid_range": rng,
        "uid_cv": std / mean if mean > 0 else 0.0,
        "uid_slope": slope,
        "uid_len": len(arr),
    }

def map_context_levels(levels):
        context_mapping = {
            "sentence": {"name": "sentence", "mode": "none"},
            "prev1": {"name": "prev1", "mode": "prev", "k": 1},
            "prev3": {"name": "prev3", "mode": "prev", "k": 3},
            "document": {"name": "document", "mode": "doc"},
            
            # sent[-L,+R] = sentence window with L sentences before, R after
            # tok[-L,+R]  = token window with L tokens before, R after
            "sent[-2,+0]": {"name": "sent[-2,+0]", "mode": "window", "window": {"type": "sent", "side": "left", "size": 2}},
            "sent[-2,+2]": {"name": "sent[-2,+2]", "mode": "window", "window": {"type": "sent", "side": "both", "size": 2}},
            "tok[-64,+0]": {"name": "tok[-64,+0]", "mode": "window", "window": {"type": "token", "side": "left", "size": 64}},
            "tok[-64,+64]": {"name": "tok[-64,+64]", "mode": "window", "window": {"type": "token", "side": "both", "size": 64}},
        }
        if levels is None:
            return list(context_mapping.values())
        return [context_mapping[level] for level in levels]

def process_surprisals(tokenizer,
                       tokens,
                       surprisals,
                       uid_unit="token"):
    """Processes surprisals to the level of the given unit.

    Args:
        tokenizer (AutoTokenizer): tokenizer used to tokenize document.
        tokens (_type_): _description_
        surprisals (_type_): _description_
        uid_unit (str, optional): _description_. Defaults to "token".
        
    TODO: Should we split by sentence in compute_surprisal before returning to make this cleaner?
    * Either way, we have to combine together to feed into model then split again
    """
    assert len(tokens) == len(surprisals), f"Dimension mismatch between inputs: surprisals ({len(surprisals)}) and tokens ({len(tokens)})"
    # skip if token-level
    if uid_unit == "token":
        return surprisals

    result_surprisals = []
    # decode and split text by sentence
    text = tokenizer.convert_tokens_to_string(tokens)
    sentences = [sent.text for sent in nlp(text).sentences]
    sentence_token_ids = [tokenizer.encode(sent, add_special_tokens=False)
                            for sent in sentences]
    sentence_token_ids_len = sum(len(sent) for sent in sentence_token_ids)
    assert sentence_token_ids_len == len(surprisals), f"Dimension mismatch between surprisals ({len(surprisals)}) and tokens ({sentence_token_ids_len}): input = {tokens}, token_ids = {sentence_token_ids}"
    
    # calculate uid units
    if uid_unit == "word":
        curr_idx = 0
        for sent_tokens in sentence_token_ids:
            sent_length = len(sent_tokens)
            curr_sentence = zip(tokens[curr_idx:curr_idx+sent_length], 
                                surprisals[curr_idx:curr_idx+sent_length])
            curr_surprisal = 0
            curr_word = ""
            for tok, surp in curr_sentence:
                if tok in tokenizer.all_special_tokens:
                    continue
                elif tok.startswith("Ġ") or tok.startswith("_"):
                    result_surprisals.append(curr_surprisal)
                    curr_surprisal = surp
                    curr_word = tok
                else:
                    curr_surprisal += surp
                    curr_word += tok
            result_surprisals.append(curr_surprisal)
            curr_idx += sent_length
    elif uid_unit == "sentence":
        # loop through surprisals and add by sentence
        surp_idx = 0
        for sent_tokens in sentence_token_ids:
            sent_surprisals = surprisals[surp_idx:surp_idx + len(sent_tokens)]
            surp_idx += len(sent_tokens)
            result_surprisals.append(sum(sent_surprisals))
    else:
        raise ValueError(f"UID Unit {uid_unit} not supported.")
    return result_surprisals

def run_uid_pipeline(
        ud_path,
        model_name="distilgpt2",
        limit_docs=3,
        limit_sents_per_doc=8,
        context_levels=None,
        generate_counterfactual=False,
        uid_level="sentence",
        uid_unit="token",
        device=None,
        output_dir=None,
        output_file=None,
        verbose=False,
        save_every=10
    ):
    print(f"Processing {ud_path}...")
    # Set up device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # metal for mac
        else:
            device = torch.device("cpu")
    
    # Generate counterfactual documents if applicable
    if generate_counterfactual:
        print("Generating counterfactual documents...", end="")
        docs = iter_counterfactual_docs(ud_path, limit_docs=limit_docs, limit_sents_per_doc=limit_sents_per_doc)
        print(" Done")
    else:
        docs = iter_ud_docs(ud_path, limit_docs=limit_docs, limit_sents_per_doc=limit_sents_per_doc)
    tokenizer, model, device = load_lm(model_name=model_name, device=device)
    docs_text = "\n".join(["\n".join(doc[1]) for doc in docs])
    unigram = UnigramLM(tokenizer)
    unigram.fit(docs_text, uid_unit=uid_unit)

    rows = []
    
    context_levels = map_context_levels(context_levels)
    # print(f"Docs to process: {len(docs)}")
    docs_pbar = tqdm(
        total=len(docs),
        desc="Processing Surprisals",
        unit="documents",
        position=0,
        leave=True
    )
    doc_idx = 0
    for doc_id, sents in docs:
        sents_pbar = tqdm(
            total=len(sents),
            desc=f"Processing {doc_id}",
            unit="sentences",
            position=1,
            leave=verbose
        )
        fact, doc_name, conv_id, conv_type = doc_id.split("::")
        for i, sent in enumerate(sents):
            if fact != 'f' and i != int(conv_id):
                sents_pbar.update(1)
                continue
            for cfg in context_levels:
                try:
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
                        sent, context, sents, i, 
                        tokenizer=tokenizer, model=model, device=device,
                        uid_level=uid_level,
                        verbose=verbose
                    )
                    surprisals = process_surprisals(tokenizer, tokens, surprisals,
                                                    uid_unit=uid_unit)
                    uni_probs, _ = unigram(tokens)
                    if len(surprisals) < 2:
                        raise ValueError(f"Too few surprisals with UID level {uid_level} and UID unit {uid_unit}!")
                    metrics = uid_metrics(surprisals, uni_probs)
                    row = {
                        "doc_id": doc_id,
                        "sent_idx": i,
                        "context": cfg["name"],
                        "sentence": sent,
                    }
                    row.update(metrics)
                    rows.append(row)
                except AssertionError as e:
                    if verbose:
                        print("Assertion error:", e)
                        print(f"Skipping {doc_id} sentence {i} for context {cfg}")
                    continue
                except ValueError as e:
                    if verbose:
                        print("Value error:", e)
                        print(f"Skipping {doc_id} sentence {i} for context {cfg}")
                    continue
            sents_pbar.update(1)
            if uid_level == 'document':
                if verbose:
                    print("Document-level analyses, skipping remaining sentences")
                break
        sents_pbar.close()
        docs_pbar.update(1)
        if doc_idx % save_every == 0:
            if verbose:
                print(f"Saving checkpoint at {doc_idx + 1} document(s)...")
            temp_df = pd.DataFrame(rows)
            temp_df.to_csv(output_dir / (output_file.stem + "_chkpt.csv"))
        doc_idx += 1
    print("Done")
    uid_df = pd.DataFrame(rows)
    return uid_df