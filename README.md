# Active Passive Switching

## Setup

Initialize the environment by using
```
uv sync
```
to install and sync all packages.

You can launch JupyterLab in our uv containerized environment with
```
uv run --with jupyter jupyter lab
```
to test and/or take a closer look at our codebase.

## Run Experiments

The entry point of this repo is `run_uid_pipeline.py`.
This script takes in arguments for the data directory, model, and various UID calculation parameters.
For example, to run the uid pipeline with the `distilgpt2` model on the .conllu files in the `data` directory, calculate UID of words at the document level, generate counterfactual documents, and save results to `outputs/cf_document_uid.csv`, run the following:
```
python run_uid_pipeline.py "data" "distilgpt2" --uid_unit="word" --uid_level="document" --generate_counterfactual --output_dir outputs --output_name "cf_document_uid.csv"
```

## Code Structure

Utility for converting between active and passive sentences is found in the `src` folder. A demo can be found in `passive_active_switch.ipynb`.

```
.
└── src/
    ├── units/
    │   ├── word.py
    │   └── sentence.py
    ├── uid.py
    ├── unigram.py
    └── utils.py
```

### Units

`word.py` contains the definition of the `Word` object, a convenient wrapper for `conllu.Token` that encodes attributes for each word. 
In our implementation, we extend `conllu.Token` to include a list of the children of each word and the word's inflection within attributes of the word.

`sentence.py` contains the definition of the `Sentence`, `PassiveSentence`, and `ActiveSentence` objects.

- `Sentence` is a wrapper for `List`, similar to `conllu.TokenList`. Initializing a sentence from a list of `Word` or `Token` objects automatically populates each word's `children` attribute with the child nodes of the word. This uses the `head` attribute of each word to determine the dependencies.
- `PassiveSentence` is an extension of `Sentence`. It stores only sentences in the passive voice with both a passive patient and passive agent. It includes functionality to depassivize the sentence, returning a deep copy of the sentence converted to active voice. Note that dependency structure is *not* updated.
- `ActiveSentence` is an extension of `Sentence`. It stores active sentences with canonical subject-object structure and includes `passivize()` to generate a passive counterfactual. Note that dependency structure is *not* updated.

`document.py` contains the definition of the `Document` object, a wrapper for a list of `Sentence` objects. This contains functionality to convert all passive sentences within each document into active sentences (and vice versa (TODO)) one at a time, via the `convert_all()` method.

### UID Calculations

`src/uid.py` contains the scripts for all uid calculations and the actual experiment.

`src/unigram.py` defines a simple unigram model for use in experiments.
A demo of this model's use can be found in `notebooks/UID.ipynb`.

### Surprisal Visualization

`src/utils.py` includes `plot_token_surprisal(tokens, surprisals, ...)` for quick token-by-token surprisal plots.
