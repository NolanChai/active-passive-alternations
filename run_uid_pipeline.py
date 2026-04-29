from src.utils import *
from src.uid import run_uid_pipeline
import argparse
from pathlib import Path
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

def main():
    # Required
    parser = argparse.ArgumentParser(description='Run Active/Passive switch script and UID calculation scripts on a given UD corpus.')
    parser.add_argument("data_dir", type=str, help="Path to folder containing .conllu files to process.")
    parser.add_argument("model",type=str, help="Model to use for surprisal calculations.")
    # Optional
    parser.add_argument("--context", "-c", type=str, default=None, help="(Optional) The context level for UID calculation. Choose between sentence, prev1, prev3, document, sent[-2,+0], sent[-2,+2], tok[-64,+0], or tok[-64,+64]. Defaults to all.")
    parser.add_argument("--generate_counterfactual", "-cf", action="store_true", help="Include to generate counterfactual documents to compare to.")    
    parser.add_argument("--limit_docs", type=int, default=None, help="(Optional) The number of documents to process.")
    parser.add_argument("--limit_sents_per_doc", type=int, default=None, help="(Optional) The number of sentences per document to process.")
    parser.add_argument("--uid_level", type=str, default="sentence", help="(Optional) Linguistic unit for which UID is analyzed. Choose between 'sentence' (default), 'document', or '(-a, +b)', in which a tokens before and b tokens after the target sentence will be analyzed.")
    parser.add_argument("--uid_unit", type=str, default="token", help="(Optional) Smallest linguistic unit for which surprisal is calculated. Choose between 'token' (default), 'word', or 'sentence'.")
    parser.add_argument("--device", type=str, default=None, help="(Optional) Override device used.")
    parser.add_argument("--output_dir", type=str, default=None, help="(Optional) Output directory")
    parser.add_argument("--output_name", type=str, default="passives_uid_calcs.csv", help="(Optional) Name of output file")
    parser.add_argument("--verbose", action="store_true", help="Set verbosity")
    
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

    # Paths
    UD_paths = Path(args.data_dir).glob('*.conllu')
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"{output_dir} does not exist. Making one for you...")
        output_dir.mkdir()
    output_file = Path(args.output_name).with_suffix(".csv")
    output_filepath = output_dir / output_file
    
    # Set up device
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # metal for mac
        else:
            device = torch.device("cpu")
    tokenizer, model, _ = load_lm(model_name=args.model, device=device)
    uid_dfs = []
    for UD_path in UD_paths:
        try:
            uid_df = run_uid_pipeline(
                ud_path=UD_path,
                model=model, 
                tokenizer=tokenizer,
                limit_docs=args.limit_docs,
                limit_sents_per_doc=args.limit_sents_per_doc,
                context_levels=args.context,
                generate_counterfactual=args.generate_counterfactual,
                uid_level=args.uid_level,
                uid_unit=args.uid_unit,
                device=device,
                output_dir=output_dir,
                output_file=output_file,
                verbose=args.verbose
            )
        except Exception as e:
            print(f"Uncaught error in file {UD_path}:")
            print(e)
            print("Skipping...")
            continue
        uid_dfs.append(uid_df)
    print(f"{len(uid_dfs)}/{len(UD_paths)} files successfully processed.")
    if len(uid_dfs) == 0:
        print("Please check files.")
        return
    uid_dfs = pd.concat(uid_dfs)
    uid_dfs.to_csv(output_filepath)
    
if __name__ == "__main__":
    main()