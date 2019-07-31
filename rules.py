""" File which has implementation of all the rules"""
from typing import Union

# spaCy imports
import spacy
from sklearn.metrics import confusion_matrix
from spacy.matcher import Matcher


class Rule:
    """Abstract class for common functionality of rules"""

    def __init__(self, nlp: spacy, verbose: bool = False):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab)
        self.counter_run = 0
        self.counter_found = 0
        self.counter_applied = 0
        self.verbose = verbose

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float) -> (bool, Union[str, spacy.tokens.doc.Doc]):
        """
            Apply this rule given this sequence
        :param sequence: a spacy doc of a string (preprocessed)
        :param probability: a float
        :return:
        """
        raise NotImplementedError(f"Class {type(self)} did not implement fn apply")

    def detect(self, sequence: spacy.tokens.doc.Doc) -> bool:
        """
            Find if this rule is applied.
            @TODO: do we return a span instead?
        :param sequence: a spacy doc of a string (preprocessed)
        :return: bool @TODO - decide later.
        """
        raise NotImplementedError(f"Class {type(self)} did not implement fn detect")

    @staticmethod
    def examples():
        raise NotImplementedError("Examples are not specified for this class")

    def test(self):
        """ Go through pos and neg examples, and find what changed and what didn't. """
        print(f" -- Testing {type(self)} --")

        pos_ex, neg_ex = self.examples()
        y_true = [1 for _ in pos_ex] + [0 for _ in neg_ex]
        y_pred = []

        # Documentize these fuckers
        pos_ex = [self.nlp(doc) for doc in pos_ex]
        neg_ex = [self.nlp(doc) for doc in neg_ex]

        for doc in pos_ex + neg_ex:
            changed, op = ruletwo.apply(doc)
            y_pred.append(1 if changed and op.text != doc.text else 0)

            if self.verbose:
                print("BEFORE:", doc)
                print("AFTER :", op)
                print('---')

        for doc in neg_ex:
            changed, op = ruletwo.apply(doc)
            y_pred.append(1 if changed else 0)

            if self.verbose:
                print("BEFORE:", doc)
                print("AFTER :", op)
                print('---')

        print(y_true)
        print(y_pred)
        print(confusion_matrix(y_true, y_pred))


class RuleOne(Rule):
    """
        **Transformation**:
            What VERB -> What's

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rulesfor Debugging NLP Models

        **Examples**
            before: What is a Hauptlied
            after: What's a Hauplied

        **Comment**
            Original intended task was `Machine Comprehension`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pattern = [{'LOWER': 'what'}, {'POS': 'VERB'}]
        self.matcher.add("RuleOne", None, self.pattern)

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> Union[str, spacy.tokens.doc.Doc]:
        """See base class"""
        matches = self.matcher(sequence)
        altered_seq = ''

        if not matches:
            if self.verbose: print("RuleOne not applied.")
            return False, sequence

        old_end_id = 0
        for match_id, start_id, end_id in matches:
            if start_id == 0:
                altered_seq += "What's "
                old_end_id = end_id
            else:
                altered_seq += sequence[old_end_id:start_id].text
                altered_seq += " What's "
                old_end_id = end_id
        altered_seq += sequence[old_end_id:].text

        return True, self.nlp(altered_seq)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += ["what is the meaning of life?",
                   "Well I've been in the desert on a horse with no name. It feels good to be out of the shade. "
                   "What is life even? In the desert no one can remember your name.",
                   "This is a first sentence. What is a second sentence? This is the third sentence. What are fourth sentences?",
                   "What works for you?"]
        neg_ex += ["which are reptiles?",
                   "what color is UK buses?",
                   "What's the meaning of life?"]
        return pos_ex, neg_ex


class RuleTwo(Rule):
    """
        **Transformation**:
            What NOUN -> Which NOUN

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rulesfor Debugging NLP Models

        **Examples**
            before: What health problem did Tesla have?
            after: Which health problem did Tesla have?

        **Comment**
            Original intended task was `Machine Comprehension/Visual Question Answering`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pattern = [{'LOWER': 'what'}, {'POS': 'NOUN'}]
        self.matcher.add("RuleTwo", None, self.pattern)

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> Union[str, spacy.tokens.doc.Doc]:
        """See base class"""
        applied = False
        matches = self.matcher(sequence)
        altered_seq = ''

        old_end_id = 0
        for match_id, start_id, end_id in matches:
            applied = True
            if start_id == 0:
                altered_seq += "Which "
                old_end_id = end_id - 1
            else:
                altered_seq += sequence[old_end_id:start_id].text
                altered_seq += " Which "
                old_end_id = end_id - 1
        altered_seq += sequence[old_end_id:].text

        return (True, self.nlp(altered_seq)) if applied else (False, sequence)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            "What health issues did Tesla have?",
            "What books tell me the meaning meaning of life?",
            "Well I've been in the desert on a horse with no name. It feels good to be out of the shade. "
            "What issues is life even? In the desert no one can remember your name.",
            "This is a first sentence. What color is the second sentence? This is the third sentence. What are fourth sentences?",
            "what color is UK buses?"
        ]
        neg_ex += [
            "which are reptiles?",
            "What's the meaning of life?",
            "This is a first sentence. What is the second sentence? This is the third sentence. What are fourth sentences?"
        ]
        return pos_ex, neg_ex


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    ruletwo = RuleOne(nlp, verbose=True)

    ruletwo.test()
