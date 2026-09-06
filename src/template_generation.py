import pandas as pd
from itertools import permutations
import numpy as np
import torch
from itertools import permutations
import nltk
nltk.download("names")
from nltk.corpus import names
from pathlib import Path
from tqdm import tqdm

from uid import *
from unigram import UnigramLM

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

if __name__ == "__main__":
    template_file = "./data/templates/coherent_discourse_pairs_500.csv"
    output_dir = "."
    output_file = "uid_results.csv"

    # Load model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # metal for mac
    else:
        device = torch.device("cpu")
    print(f"Loading model on {device}...", end="")
    tokenizer, model, _ = load_lm(model_name='distilgpt2', device=device)
    print("Done.")
    
    # Load templates
    print(f"Loading templates from {template_file}...")
    templates = pd.read_csv(template_file)
    print("Done.")

    # Generate data
    print("Generating data...", end="")
    variants = []
    names_list = names.words("female.txt") + names.words("male.txt")
    np.random.seed(3)
    names_sample = np.random.choice(names_list, size=4)
    print(names_sample)
    variants = generate_data(templates, names_sample)
    print("Done.")
    for sent in np.random.choice(variants['text'], 10):
        print(" - " + sent)

    # Process results
    print("Processing results...", end="")
    uid_results = process_results(model, tokenizer, device, variants)
    print("Done.")

    print(f"Saving results to {Path(output_dir) / Path(output_file)}...", end="")
    uid_results.to_csv(Path(output_dir) / Path(output_file), index=False)
    print("Done.")

    print("Analysis complete. Results saved to uid_results.csv.")