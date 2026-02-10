from src.utils import *
from src.uid import run_uid_pipeline
import argparse
from pathlib import Path
import pandas as pd
import re

def main():
    parser = argparse.ArgumentParser(description='Run Active/Passive switch script and UID calculation scripts on a given UD corpus.')
    parser.add_argument("data_dir", type=str, help="Path to folder containing .conllu files to process.") 
    parser.add_argument("model",type=str, help="Model to use for surprisal calculations.")
    parser.add_argument("--context", "-c", type=str, default=None, help="The context level for UID calculation. Choose between sentence, prev1, prev3, document, sent[-2,+0], sent[-2,+2], tok[-64,+0], or tok[-64,+64].")
    parser.add_argument("--limit_docs", type=int, default=None, help="(Optional) The number of documents to process.")
    parser.add_argument("--limit_sents_per_doc", type=int, default=None, help="(Optional) The number of sentences per document to process.")
    parser.add_argument("--generate_counterfactual", "-cf", action="store_true", help="Include to generate counterfactual documents to compare to.")
    
    args, unk = parser.parse_known_args()
    
    # Handle unknown args and save jic
    extra_args = {}
    for arg in unk:
        # edge case handling
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Convert value to appropriate type
            if value.lower() == 'true':
                extra_args[key] = True
            elif value.lower() == 'false':
                extra_args[key] = False
            elif value.isdigit():
                extra_args[key] = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                extra_args[key] = float(value)
            else:
                extra_args[key] = value

    UD_paths = Path(args.data_dir).iterdir()
    uid_dfs = []
    for UD_path in UD_paths:
        uid_df = run_uid_pipeline(
            UD_path,
            model_name=args.model,
            limit_docs=args.limit_docs,
            limit_sents_per_doc=args.limit_sents_per_doc,
            context_levels=args.context,
            generate_counterfactual=args.generate_counterfactual
        )
        uid_dfs.append(uid_df)
    uid_dfs = pd.concat(uid_dfs)
    uid_dfs.to_csv("passives_uid_calcs.csv")
    

if __name__ == "__main__":
    main()