import pyinflect
from pyinflect import getAllInflections, getInflection
import conllu
from conllu import Token, TokenList
from .word import Word
from .sentence import Sentence, PassiveSentence

class Document(list[Sentence]):
    """
    Wrapper class for a list of sentences, representing a document.
    """
    def __init__(self, sentences: list[list]) -> None:
        # Automatically cast passive sentences to PassiveSentence
        sentence_list = []
        for s in sentences:
            try:
                sentence_list.append(PassiveSentence(s))
            except ValueError:
                sentence_list.append(Sentence(s))
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
        self.num_actives = len(self) - self.num_passives # redefine once actives are more defined
    
    def convert_all(self):
        """
        Converts each passive sentence -> active and each active sentence -> 
        passive, one at a time. Results in num_passives + num_actives counterfactual documents.
        Returns:
            list[Document]: List of all counterfactual documents
        """
        result = []
        passive_indices = [-1]
        for _ in range(self.num_passives):
            counterfactual_doc = []
            # append all sentences up to previously found passive
            for s in self[:passive_indices[-1] + 1]:
                counterfactual_doc.append(s.deep_copy())
            # append until we find a passive; depassivize and append
            for i, s in enumerate(self[passive_indices[-1] + 1:]):
                if isinstance(s, PassiveSentence):
                    counterfactual_doc.append(s.depassivize())
                    # update index of previous passive
                    passive_indices.append(passive_indices[-1] + i + 1)
                    break
                else:
                    counterfactual_doc.append(s.deep_copy())
            # append remaining sentences
            for s in self[passive_indices[-1] + 1:]:
                counterfactual_doc.append(s.deep_copy())
            # print(len(counterfactual_doc))
            # print(counterfactual_doc[1])
            # print(counterfactual_doc[1].metadata)
            result.append(Document(counterfactual_doc))
        return zip(result, passive_indices[1:])
    
    def format_doc(self):
        """Formats text in the document from sentences, putting a newline at paragraph boundaries.

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
        return self.num_passives > 0

    def has_active(self) -> bool:
        return self.num_actives > 0