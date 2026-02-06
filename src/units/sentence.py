import pyinflect
from pyinflect import getAllInflections, getInflection
import conllu
from conllu import Token, TokenList
from .word import Word

def switch_pronoun(const):
    """ Switch relevant pronouns from subject to object.

    Args:
        const (list): list of Word objects to process
    """
    pron_key = {
        "i": "me",
        "he": "him",
        "she": "her",
        "we": "us",
        "they": "them",
        "me": "I",
        "him": "he",
        "her": "she",
        "us": "we",
        "them": "they"
    }
    for w in const:
        # print(w['form'], w['deprel'])
        if w['deprel'] in ['obl:agent', 'nsubj:pass'] and w['form'].lower() in pron_key:
            w['form'] = pron_key[w['form'].lower()]

def build_subtree(root, upos: list = None, deprel: list = None):
    """return all words under the root in the dependency tree, subject to filters.

    Args:
        root (Word): The root word of the subtree to build.

    Raises:
        ValueError: If root is not a Word object.

    Returns:
        list[Word]: The list of Word objects in the subtree, ordered by ID.
    """
    if not isinstance(root, Word):
        raise ValueError("root must be a Word object")

    subtree = []
    to_visit = [root]
    use_filter = upos is not None or deprel is not None
    upos_set = set(upos or [])
    deprel_set = set(deprel or [])

    while to_visit:
        node = to_visit.pop()
        subtree.append(node)
        if use_filter:
            to_visit += [child for child in node['children']
                        if child['upos'] in upos_set
                        or child['deprel'] in deprel_set]
        else:
            to_visit += node['children']
    subtree = sorted(subtree, key=lambda w: w['id'])
    return Sentence(subtree)

class Sentence(list[Word]):
    """
    Wrapper class for a list of Word objects to represent a sentence.
    """
    def __init__(self, tokens: list[dict]) -> None:
        super().__init__([Word(token_dict=t) for t in tokens
                          if t['lemma'] != '_'])
        
        try:
            self.metadata = tokens.metadata
        except AttributeError:
            self.metadata = None
        
        self.root = next((word for word in self if word['head'] == 0), None)
        
        if isinstance(tokens, TokenList):
            self.text = tokens.metadata['text']
        elif isinstance(tokens, Sentence):
            self.text = tokens.text
        else:
            self.text = ' '.join([word['form'] for word in self])
            
        self.is_passive = (any(word['deprel'] == 'obl:agent' for word in self) 
                           and any(word['deprel'] == 'nsubj:pass' for word in self))
        
        self.populate_children()
        
    def populate_children(self):
        """Populate the children attribute for each Word in the sentence."""
        id_to_word = {word['id']: word for word in self}
        for word in self:
            head_id = word['head']
            if head_id:
                if head_id not in id_to_word:
                    continue
                head_word = id_to_word[head_id]
                head_word['children'].append(word)
                
    def deep_copy(self):
        """Creates a deep copy of the sentence.
        
        Returns:
            Sentence: Deep copy with same words.
        """
        s_list = [w.deep_copy() for w in self]
        try:
            result = PassiveSentence(s_list)
        except ValueError:
            result = Sentence(s_list)
        if self.metadata is None:
            result.metadata = None
        else:
            result.metadata = {key: value for key, value in self.metadata.items()}
        return result

    def __str__(self):
        return self.text
                
class PassiveSentence(Sentence):
    def __init__(self, tokens: list[dict]) -> None:
        super().__init__(tokens)
        if not self.is_passive:
            raise ValueError("The provided sentence is not passive (missing either nsubj:pass or obl:agent).")
        
        # Extract core
        self.passive_subject, self.passive_subject_word = self.find_pass_subj()
        self.verb, self.verb_word = self.find_verb()
        self.auxpass = next(filter(lambda w: w['deprel'] == 'aux:pass', self.verb), None)
        if self.auxpass is None:
            raise ValueError("No auxiliary passive verb found.")
        self.agent, self.agent_word = self.find_pass_agent()
        if self.passive_subject is None or self.passive_subject_word is None:
            raise ValueError("No passive subject found in the sentence.")
        if self.verb is None or self.verb_word is None:
            raise ValueError("No verb found in the sentence.")
        if self.agent is None or self.agent_word is None:
            raise ValueError("No passive agent found in the sentence.")
    
    def depassivize(self):
        """
        Convert the passive sentence to active voice. Note that dependencies in
        the resulting sentence will not be accurate.

        Returns:
            Sentence: a deep copy of the sentence, converted to active voice. 
        """
        words = [w.deep_copy() for w in self]
        drop_by_ids = set()
        for w in self:
            if w['deprel'] == 'case' and w['form'].lower() == 'by':
                head = next((t for t in self if t['id'] == w['head']), None)
                if head and head['deprel'] == 'advcl' and head['head'] == self.verb_word['id']:
                    if any(c['deprel'] == 'mark' and c['form'].lower() == 'than' for c in head['children']):
                        drop_by_ids.add(w['id'])
        id_to_idx = {w['id']: i for i, w in enumerate(words)}

        def span_from_ids(ids):
            idxs = [id_to_idx[i] for i in ids if i in id_to_idx]
            return (min(idxs), max(idxs))

        verb_ids = [w['id'] for w in self.verb]
        subj_ids = [w['id'] for w in self.passive_subject]
        agent_ids = [w['id'] for w in self.agent]

        # Get indices for passive components
        verb_span = span_from_ids(verb_ids)
        subj_span = span_from_ids(subj_ids)
        agent_span = span_from_ids(agent_ids)

        # reinflect/adjust words
        verb_const = self.activize_verb()
        subj_const = self.activize_subj()
        agent_const = self.activize_agent()
        prefix_ids = set(w['id'] for w in words[0:subj_span[0]])
        verb_const = [w for w in verb_const if w['id'] not in prefix_ids]

        # Reorder sentence
        # account for relative clauses
        if self.passive_subject_word['feats'].get('PronType', None) == 'Rel':
            activized_sentence = (words[0:subj_span[0]]
                + subj_const
                + agent_const
                + words[subj_span[1]+1:verb_span[0]]
                + verb_const
                + words[verb_span[1]+1:agent_span[0]]
                + words[agent_span[1]+1:]
            )
        else:
            activized_sentence = (words[0:subj_span[0]]
                + agent_const
                + words[subj_span[1]+1:verb_span[0]]
                + verb_const
                + subj_const
                + words[verb_span[1]+1:agent_span[0]]
                + words[agent_span[1]+1:]
            )
        if drop_by_ids:
            activized_sentence = [w for w in activized_sentence if w['id'] not in drop_by_ids]
        activized_sentence = [w.deep_copy() for w in activized_sentence]
        activized_sentence[0]['form'] = activized_sentence[0]['form'].title()
        activized_sentence = Sentence(activized_sentence)
        if self.metadata is None:
            activized_sentence.metadata = None
        else:
            activized_sentence.metadata = {key: value for key, value in self.metadata.items()}
            activized_sentence.metadata['text'] = activized_sentence.text
        return activized_sentence

    def activize_verb(self, verbose=False):
        """
        Find the active form of the verb constituent in the sentence.

        Raises:
            ValueError: if no auxiliary passive verb is found.

        Returns:
            List: The active form of the verb constituent.
        """
        # deep copy to not modify original
        verb_const = [w.deep_copy() for w in self.verb]
        verb_word = next(filter(lambda w: w['id'] == self.verb_word['id'], verb_const), None)
        
        auxpass = next(filter(lambda w: w['deprel'] == 'aux:pass', verb_const), None)
        if auxpass is None:
            raise ValueError("No auxiliary passive verb found.")
        verb_const.remove(auxpass)

        # detect clausal negation tied to verb or its auxiliaries
        aux_ids = {w['id'] for w in self.verb if w['deprel'] in ['aux', 'aux:pass']}
        neg_tokens = []
        for w in self:
            feats = w.get('feats') or {}
            if (w['lemma'] == 'not' or w['form'].lower() in ["not", "n't"]) and feats.get('Polarity', None) == 'Neg':
                if w['head'] == self.verb_word['id'] or w['head'] in aux_ids:
                    neg_tokens.append(w)
        has_focus_neg = any(any(o['lemma'] == 'only' and (o['head'] == n['head'] or o['head'] == n['id']) for o in self)
                            for n in neg_tokens)
        if neg_tokens:
            verb_ids = set(w['id'] for w in verb_const)
            for n in neg_tokens:
                if n['id'] not in verb_ids:
                    verb_const.append(n.deep_copy())

        # reinflect main verb and auxiliaries
        auxpass_infl = auxpass['inflection']
        agent_infl = self.agent_word['inflection']
        
        # Determine new inflection for main verb
        verb_word['form'] = getInflection(verb_word['lemma'], auxpass_infl)[0]
        # add do-support for clausal negation when no other auxiliary is present
        if neg_tokens and not has_focus_neg:
            has_other_aux = any(w['upos'] == 'AUX' or w['xpos'] == 'MD' for w in verb_const)
            if not has_other_aux:
                auxpass['lemma'] = 'do'
                auxpass['upos'] = 'AUX'
                auxpass['deprel'] = 'aux'
                auxpass['xpos'] = auxpass_infl
                auxpass['inflection'] = auxpass_infl
                do_form = getInflection('do', auxpass_infl)
                if do_form:
                    auxpass['form'] = do_form[0]
                else:
                    auxpass['form'] = 'do'
                base_form = getInflection(verb_word['lemma'], 'VB')
                verb_word['form'] = base_form[0] if base_form else verb_word['lemma']
                verb_const.append(auxpass)

        # sort verb_const by ID so do-support is inflected instead of main verb
        verb_const = sorted(verb_const, key=lambda w: w['id'])

        # Inflect first auxiliary verb/main verb according to agent
        verb_person_key = {
            'me': '1',
            'you': '2'
        }
        verb_tense_key = {
            'VB': 'present',
            'VBD': 'past',
            'VBN': 'past',
            'VBP': 'present',
            'VBZ': 'present'
        }
        verb_infl_key = {
            '1Spast': 'VBD',
            '2Spast': 'VBD',
            '3Spast': 'VBD',
            '1Ppast': 'VBD',
            '2Ppast': 'VBD',
            '3Ppast': 'VBD',
            '1Spresent': 'VBP',
            '2Spresent': 'VBP',
            '3Spresent': 'VBZ',
            '1Ppresent': 'VBP',
            '2Ppresent': 'VBP',
            '3Ppresent': 'VBP',
        }
        for w in verb_const:
            if w['upos'] in ['VERB', 'AUX']:
                if w['xpos'] == 'MD':
                    verb_infl = 'MD'
                    break
                verb_person = verb_person_key.get(self.agent_word['lemma'].lower(), 3)
                verb_number = 'S' if agent_infl == 'NN' else 'P'
                if w['upos'] == 'VERB':
                    verb_tense = verb_tense_key.get(auxpass['inflection'], 'present')
                else:
                    verb_tense = verb_tense_key.get(w['inflection'], 'present')
                verb_infl = verb_infl_key[f'{verb_person}{verb_number}{verb_tense}']
                inflection_idx = -1 if verb_person == '2' or verb_number == 'P' else 0
                # 2/6/2026 - avoid archaic do-support forms like "didst"/"dost"
                if w['lemma'] == 'do':
                    if verb_infl == 'VBD':
                        w['form'] = 'did'
                    elif verb_infl == 'VBZ':
                        w['form'] = 'does'
                    else:
                        w['form'] = 'do'
                else:
                    w['form'] = getInflection(w['lemma'], verb_infl)[inflection_idx]
                
                # print("pnt:", f'{verb_person}{verb_number}{verb_tense}')
                break
        if verbose:
            print("auxp:", auxpass_infl, 
                "| agent_infl:", agent_infl, 
                "| verb_infl:", verb_infl, 
                "| verb:", verb_word['form'])
            print(self.agent_word['form'], ' '.join(w['form'] for w in verb_const))
        return verb_const
    
    def activize_subj(self):
        """
        Find the active form of the passive subject constituent in the sentence.
        Converts pronouns from subject -> object.

        Returns:
            List: The active form of the subject constituent.
        """
        subj_const = [w.deep_copy() for w in self.passive_subject]
        switch_pronoun(subj_const)
        
        # Turn form to lowercase if not proper noun
        if subj_const[0]['upos'] != 'PROPN':
            subj_const[0]['form'] = subj_const[0]['form'].lower()
        return subj_const
    
    def activize_agent(self):
        """
        Find the active form of the agent constituent in the sentence.
        Removes 'by' and converts pronouns from object -> subject.

        Returns:
            List: The active form of the agent constituent.
        """
        # remove 'by' (including coordinated agent heads)
        agent_heads = {self.agent_word['id']}
        added = True
        while added:
            added = False
            for w in self.agent:
                if w['deprel'] == 'conj' and w['head'] in agent_heads and w['id'] not in agent_heads:
                    agent_heads.add(w['id'])
                    added = True
        agent_const = list(filter(lambda w: not (
                                  w['deprel'] == 'case'
                                  and w['form'].lower() == 'by'
                                  and w['head'] in agent_heads), self.agent))
        agent_const = [w.deep_copy() for w in agent_const]
        switch_pronoun(agent_const)
        return agent_const

    def find_pass_subj(self):
        """
        Extract the passive subject + constituent from this sentence.
        
        Returns:
            tuple: (list of Word objects in the passive subject subtree, Word object of the passive subject)
        """

        # Passive subject takes on the 'nsubj:pass' deprel tag
        # TODO: Support multiple passive subjects?
        subj = next(filter(lambda w: w['deprel'] == "nsubj:pass", self), None)
        if subj is None:
            return None, None

        subj_subtree = build_subtree(subj)
        return subj_subtree, subj

    def find_verb(self):
        """
        Extract the verb + constituent from this sentence.
        
        Returns:
            tuple: (list of Word objects in the verb subtree, Word object of the verb)
        """
        # Get the verb that is the head of the passive subject
        verb = None
        if self.passive_subject_word['head'] > 0:
            verb = next(filter(lambda w: w['id'] == self.passive_subject_word['head'], self), None)
        if verb is None or verb['upos'] != "VERB":
            return None, None
        verb_subtree = build_subtree(verb,
                                upos=['ADV'],
                                deprel=['aux', 'aux:pass']) # pass aux will be removed later
        
        # remove adverbs that come after the verb
        verb_subtree = Sentence(list(filter(lambda w: w['upos'] != 'ADV' or w['id'] < verb['id'], verb_subtree)))
        return verb_subtree, verb
    
    def find_pass_agent(self):
        """
        Extracts the passive agent from this sentence.

        Returns:
            tuple: (list of Word objects in the passive agent subtree, Word object of the passive agent)
        """
        # Agent has the deprel tag 'obl:agent' and is attached to the same verb
        agent = next(
            filter(
                lambda w: w['deprel'] == "obl:agent"
                    and w in self.verb_word['children'],
                self
                ),
            None
        )
        # In case agent isn't found that way, look for 'obl' tag and a "by" phrase
        if agent is None:
            for w in self:
                if w['deprel'] == "obl" and w in self.verb_word['children']:
                    case_children = [c for c in w['children'] if c['deprel'] == "case"]
                    if any(c['form'].lower() == "by" for c in case_children):
                        agent = w
                        break
        if agent is None:
            return None, None

        # extract subtree
        agent_subtree = build_subtree(agent)

        # ids -> words
        return agent_subtree, agent
