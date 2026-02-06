# Changelog
General use-case changelog to track commits and changes
### 2/6/2026 Fixes
#### Quotes fix
* There was an issue where closing quotes and parentheses would sometimes be attached to the agent head as `punct`, so it gets pulled into the agent subtree and moves to the front when you swap constituents. Specifically:
  * Changed `depassivize()` so that it excludes quote punctuation that isn’t adjacent to the agent core, so quotes tied to the passive subject stay put.
  * Changed `detokenize()` to add `'` and `‘` to avoid the dog spacing issue
  * Referring to this case: 
  > Thus, the sound of the word dog in English is connected to the concept ‘dog’ by historical accident and not by any natural connection; roughly the same concept is just as well denoted in French by chien, in German by hund, and in Japanese by inu.

  Now correctly results in:
  > Thus, historical accident and not any natural connection connects the sound of the word dog in English to the concept ‘dog’; roughly the same concept is just as well denoted in French by chien, denoted in German by hund, and denoted in Japanese by inu.

#### Closing punctuation spacing fix
* Closing punctuation would sometimes lose the space after it when followed by a word, resulting in cases like `)determines` or `”)initiates`.
  * Changed `detokenize()` so `SpaceAfter=No` is ignored for closing punctuation

#### Minor weird double adverb fix
* Added `cc` to verb subtree filter so coordinated adverbs keep their conjunction