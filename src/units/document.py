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
        assert all(map(lambda s: "newdoc id" not in s.metadata, self[1:])), "Sentence list should only contain one document"
        
        self.doc_id = self[0].metadata["newdoc id"]
        
        self.metadata = dict()
        for key, value in self[0].metadata.items():
            if "meta" in key:
                attr_name = key.split('::')[-1].strip()
                self.metadata[attr_name] = value
        
        self.text = self.format_doc()
    
    def convert_all(self):
        """
        Converts each passive sentence -> active and each active sentence -> passive.
        Returns:
            list[Document]: List of all counterfactual documents
        """
        return
    
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
        