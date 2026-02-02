# Active Passive Switching

## Setup

Initialize by using
```
uv sync
```
to install and sync all packages.

You can launch Jupyter Notebook in a uv containerized environment with
```
uv run --with jupyter jupyter lab
```

## Code Structure

Utility for converting between active and passive sentences is found in the `src` folder. A demo can be found in `passive_active_switch.ipynb`.

```
.
└── src/
    ├── units/
    │   ├── word.py
    │   └── sentence.py
    └── utils.py
```

`word.py` contains the definition of the `Word` object, a convenient wrapper for `conllu.Token` that encodes attributes for each word. 
In our implementation, we extend `conllu.Token` to include a list of the children of each word and the word's inflection within attributes of the word.

`sentence.py` contains the definition of the `Sentence`, `PassiveSentence`, and `ActiveSentence` objects.

- `Sentence` is a wrapper for `List`, similar to `conllu.TokenList`. Initializing a sentence from a list of `Word` or `Token` objects automatically populates each word's `children` attribute with the child nodes of the word. This uses the `head` attribute of each word to determine the dependencies.
- `PassiveSentence` is an extension of `Sentence`. It stores only sentences in the passive voice with both a passive patient and passive agent. It includes functionality to depassivize the sentence, returning a deep copy of the sentence converted to active voice. Note that dependency structure is *not* updated.
- `ActiveSentence` TODO
