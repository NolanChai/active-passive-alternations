import pandas as pd
from itertools import permutations
import numpy as np
import torch

def replace_map(string, mapping):
    for key, value in mapping.items():
        string = string.replace(key, value)
    return string

def generate_variants(template,
                      agent,
                      patient,
                      control1,
                      control2,
                      passive):
    """Generates all variants of a template given the subject, object, control
    noun, and template. Template should be in the form "<PRE> [preamble]. <AGT>
    [verb] <PNT>." for actives or "<PRE> [preamble]. <PNT> was [verb] by <AGT>."
    for passives, where angle brackets are formatted as seen and square brackets
    are replaced with their respective elements.
        
    Args:
        agent (str): The agent to use.
        patient (str): The patient to use.
        control1 (str): The first control noun to use. Placed at beginning of preamble.
        control2 (str): The second control noun to use. Placed at end of preamble.
        template (str): Template in the form shown above.
        passive (bool):
         
    Returns:
        _type_: _description_
    """
    result = []
    text = replace_map(template, {"<AGT>": agent, "<PNT>": patient, "<CTL>": control2})
    for noun in [agent, patient, control1]:
        result.append({
            "patient": patient,
            "agent": agent,
            "control": (control1, control2),
            "passive": passive,
            "subject": patient if passive else agent,
            "object": agent if passive else patient,
            "given": noun,
            "text": text.replace("<PRE>", noun)
        })
    return pd.DataFrame(result)

def generate_data(templates, names):
    variants = []
    for idx, row in templates.iterrows():
        for perm in permutations(names):
            for template in [row['passive'], row['active']]:
                result = generate_variants(template, *perm, template==row['passive'])
                result['pair_id'] = row['pair_id']
                variants.append(result)
        
    variants = pd.concat(variants, ignore_index=True)
    variants = variants.merge(templates.drop(columns=['passive', 'active']), on='pair_id', how='left')
    return variants

def process_results(model, tokenizer, device, variants, save_results_to_file=True):
    uid_results = []
    unigram = UnigramLM(tokenizer)
    unigram.fit(" ".join(variants['text']), uid_unit="word")
    for i in tqdm(range(variants.shape[0])):
        try:
            variant = variants.iloc[i]
            text = variant['text']
            preamble, sentence = text.split(". ")
            preamble += "."
            for context in [preamble, ""]:
                tokens, surprisals = compute_surprisal(sentence, 
                                context, 
                                sentences=[context, sentence], 
                                sent_idx=1, 
                                tokenizer=tokenizer, 
                                model=model,
                                device=device)
                surprisals, units = process_surprisals(tokenizer,
                                                    tokens,
                                                    surprisals,
                                                    uid_unit="word")
                uni_probs, _ = unigram(tokens)
                result = get_uid_metrics(surprisals, uni_probs)
                result['units'] = units
                result['context'] = context
                result['sentence'] = sentence
                result.update(dict(variant))
                uid_results.append(result)
            if i % 100 == 0 & save_results_to_file:
                # print(f"Processed {i} sentences")
                pd.DataFrame(uid_results).to_csv("temp_results.csv", index=False)
        except Exception as e:
            print(f"Error processing sentence: {text}")
            print(f"Error details: {e}")
            continue
    uid_results = pd.DataFrame(uid_results)
    if save_results_to_file:
        uid_results.to_csv("uid_results.csv", index=False)
    return uid_results