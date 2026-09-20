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

from src.uid import *
from src.template_generation import *
from src.unigram import UnigramLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Active/Passive sythentic data (template) generation and UID calculation scripts.')
    parser.add_argument("templates_dir", type=str, help="Path to folder containing .csv files with templates to process.")
    parser.add_argument("output_dir",type=str, help="Path to folder where output files will be saved.")
    # Optional
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