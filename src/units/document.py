import pyinflect
from pyinflect import getAllInflections, getInflection
import conllu
from conllu import Token, TokenList
from .word import Word
from .sentence import Sentence, PassiveSentence, ActiveSentence

class Document(list[Sentence]):
    """
    Wrapper class for a list of sentences, representing a document.
    """
    def __init__(self, sentences: list[list]) -> None:
        # Automatically cast passive sentences to PassiveSentence
        sentence_list = []
        for s in sentences:
            try:
                s = Sentence(s)
                if s.is_passive:
                    sentence_list.append(PassiveSentence(s))
                else:
                    sentence_list.append(ActiveSentence(s))
            except ValueError:
                sentence_list.append(s)
        super().__init__(sentence_list)
        
        # First sentence only should contain document metadata
        assert self[0].metadata.get("newdoc id", None), "First sentence should contain document metadata"
        for s in self[1:]:
            assert "newdoc id" not in s.metadata, "Sentence list should only contain one document"
        
        self.doc_id = self[0].metadata["newdoc id"]
        
        self.metadata = dict()
        for key, value in self[0].metadata.items():
            if "meta" in key:
                attr_name = key.split('::')[-1].strip()
                self.metadata[attr_name] = value
        
        self.text = self.format_doc()
        
        self.num_passives = sum(map(lambda s: isinstance(s, PassiveSentence), self))
        self.num_actives = sum(map(lambda s: isinstance(s, ActiveSentence), self))
    
    def convert_all(self):
        """
        Converts each convertible passive sentence -> active and each active sentence -> 
        passive, one at a time. Results in num_passives + num_actives counterfactual documents.
        Returns:
            List[tuple(Document, int, str)]: List of (Document, idx, conversion) pairs, where idx is the index of the converted sentence and conversion is the conversion applied.
        """
        result = []
        target_indices = [-1]
        conversions = []
        for _ in range(self.num_passives + self.num_actives):
            counterfactual_doc = []
            # append all sentences up to previously found target
            for s in self[:target_indices[-1] + 1]:
                counterfactual_doc.append(s.deep_copy())
            # append until we find a convertable sentence; convert and append
            try:
                for i, s in enumerate(self[target_indices[-1] + 1:]):
                    if isinstance(s, PassiveSentence):
                        counterfactual_doc.append(s.depassivize())
                        # update index of previous passive
                        target_indices.append(target_indices[-1] + i + 1)
                        conversions.append("p>a")
                        break
                    elif isinstance(s, ActiveSentence):
                        counterfactual_doc.append(s.passivize())
                        target_indices.append(target_indices[-1] + i + 1)
                        conversions.append("a>p")
                        break
                    else:
                        counterfactual_doc.append(s.deep_copy())
            except Exception as e:
                print(f"Error on doc {self.doc_id}, sentence {i}")
                print(e)
                target_indices.append(target_indices[-1] + i + 1)
                continue
            # append remaining sentences
            for s in self[target_indices[-1] + 1:]:
                counterfactual_doc.append(s.deep_copy())
            result.append(Document(counterfactual_doc))
            
        return list(zip(result, target_indices[1:], conversions))
    
    def format_doc(self):
        """Formats text in the document from sentences, putting a newline at 
        paragraph boundaries.
        Returns:
            str: formatted text
        """
        result = ""
        for s in self:
            result += ' '
            if "newpar" in s.metadata:
                result += '\n'
            result += s.text
        result = result.strip()
        return result
    
    def __str__(self):
        return str(self.text)

    def has_passive(self) -> bool:
        """
        Returns:
            bool: whether or not there is at least one passive sentence in the document.
        """
        return self.num_passives > 0

    def has_active(self) -> bool:
        """
        Returns:
            bool: whether or not there is at least one active sentence in the document.
        """
        return self.num_actives > 0