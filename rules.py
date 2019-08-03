""" File which has implementation of all the rules"""
from typing import Union
from sklearn.metrics import confusion_matrix

# spaCy imports
import spacy
from spacy.matcher import Matcher

# Local imports
import utils

class Rule:
    """Abstract class for common functionality of rules"""

    def __init__(self, nlp: spacy, verbose: bool = False):
        self.nlp = nlp
        self.matcher = Matcher(self.nlp.vocab)
        self.counter_run = 0
        self.counter_found = 0
        self.counter_applied = 0
        self.verbose = verbose

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> (bool, Union[str, spacy.tokens.doc.Doc]):
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
        """
            A bunch of positive and negative examples for this rule.
            Positive -> the rule can be applied
            Negative -> the rule should not be applied

        :return:
        """
        raise NotImplementedError("Examples are not specified for this class")

    def test(self):
        """ Go through pos and neg examples, and find what changed and what didn't. """
        print(f" -- Testing {type(self)} --")

        pos_ex, neg_ex = self.examples()
        y_true = [1 for _ in pos_ex] + [0 for _ in neg_ex]
        y_pred = []

        # Document-ize these
        pos_ex = [self.nlp(doc) for doc in pos_ex]
        neg_ex = [self.nlp(doc) for doc in neg_ex]

        for doc in pos_ex + neg_ex:
            changed, op = self.apply(doc)
            y_pred.append(1 if changed and op.text != doc.text else 0)

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

    @staticmethod
    def _can_be_applied_(sequence) -> bool:
        """Internal fn which tells whether this rule can be applied on this sentence/sequence"""
        return sequence[0].lower_ == 'what' and sequence[1].lower_ in ['is', 'are', 'was', 'were']

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> Union[str, spacy.tokens.doc.Doc]:
        """See base class"""
        alt_sequence = ''
        applied = False

        for sentence in sequence.sents:
            if self._can_be_applied_(sentence):
                # If this is not the first sentence, check if the token _before_ this had space and add accordingly
                # if sentence[0].i != 0:
                #     alt_sequence += utils.need_space_after_(token=sequence[sentence[0].i-1])
                alt_sequence += "What's"
                alt_sequence += utils.need_space_after_(token=sentence[1])
                alt_sequence += sentence[2:].text_with_ws
                applied = True
            else:
                alt_sequence += sentence.text_with_ws
        return (applied, self.nlp(alt_sequence)) if applied else (False, sequence)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            "what is the meaning of life?",
            "Well I've been in the desert on a horse with no name. It feels good to be out of the shade. "
            "What is life even? In the desert no one can remember your name.",
            "This is a first sentence. What is a second sentence? This is the third sentence. What are fourth sentences?"
        ]
        neg_ex += [
            "which are reptiles?",
            "what color is UK buses?",
            "What's the meaning of life?",
            "What works for you?"
        ]
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
            "What's color of Indian taxis?",
            "which are reptiles?",
            "What's the meaning of life?",
            "This is a first sentence. What is the second sentence? This is the third sentence. What are fourth sentences?"
        ]
        return pos_ex, neg_ex


class RuleThree(Rule):
    """
        **Transformation**:
            What VERB -> So what VERB

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rulesfor Debugging NLP Models

        **Examples**
            before: What was Gandhi's work called?
            after: So what was Gandhi's work called?

        **Comment**
            Original intended task was `Machine Comprehension/Visual Question Answering`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pattern = [{'LOWER': 'what'}, {'POS': 'VERB'}]
        self.matcher.add("RuleTwo", None, self.pattern)

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> (bool, Union[str, spacy.tokens.doc.Doc]):
        """See base class"""
        applied = False
        matches = self.matcher(sequence)
        alt_sequence = ''

        '''
            Rule application logic
                -> if 'what VERB' appears in seq
                    -> if 'so' does not appear before it
                    
            @TODO: fix 
                BEFORE: What's the point of working like this?
                AFTER : So what 's the point of working like this? (EXTRA SPACE)
        '''

        old_end_id = 0
        for match_id, start_id, end_id in matches:
            if start_id == 0:
                applied = True
                alt_sequence += "So what"
                alt_sequence += utils.need_space_after_(token=sequence[start_id])
                old_end_id = end_id - 1
            else:
                if sequence[start_id-1].lower_ == 'so':
                    continue
                applied = True
                # Add existing sequence to the alt_seq
                alt_sequence += sequence[old_end_id:start_id].text_with_ws

                # If we're in a new sentence, add "So" else "so"
                alt_sequence += "So what" if sequence[start_id-1].is_sent_start else "so what"
                
                alt_sequence += utils.need_space_after_(token=sequence[start_id])
                old_end_id = end_id - 1
        alt_sequence += sequence[old_end_id:].text

        return (True, self.nlp(alt_sequence)) if applied else (False, sequence)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            "What was Gandhi's work called?",
            "What's the point of working like this?",
            "what is the meaning of life?",
            "It is the desert. What is life even? In the desert no one can remember your name.",
            "This is a first sentence. What is a second sentence? This is the third sentence. What are fourth sentences?"
        ]
        neg_ex += [
            "So what if you think what runs will keep on running?",
            "what if you think what runs will keep on running?",
            "So what was Gandhi's work called?",
            "Which is the meaning of life?",
            "It is the desert. So what is life even?",
            "What Gaurav said.",
            "What Barack Obama ate this week?"
        ]
        return pos_ex, neg_ex


class RuleFour(Rule):
    """
        **Transformation**:
            What VBD -> And what VBD

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rulesfor Debugging NLP Models

        **Examples**
            before: What was Kenneth Sweezy's job?
            after: And what was Kenneth Sweezy's job?

        **Comment**
            Original intended task was `Machine Comprehension/Visual Question Answering`
            Only applicable when `what` is at sentence start.
    """

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> (bool, Union[str, spacy.tokens.doc.Doc]):
        """See base class"""
        applied = False
        alt_sequence = ''

        for sentence in sequence.sents:
            if sentence[0].lower_ == 'what' and sentence[1].tag_ == 'VBD':
                # Apply Rule
                alt_sequence += 'And what'
                alt_sequence += utils.need_space_after_(sentence[0])
                alt_sequence += sentence[1:].text_with_ws

                applied = True
            else:
                alt_sequence += sentence.text_with_ws

        return (True, self.nlp(alt_sequence)) if applied else (False, sequence)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            "What was Gandhi's work called?",
            "what was it you said was the meaning of life?",
            "It is the desert. What used to be life even? In the desert no one can remember your name.",
            "This is a first sentence. What was the second sentence? This is what was supposed to be the third one?"
        ]
        neg_ex += [
            "And what was that you said?",
            "What's the point of working like this?",
            "what if you think what runs will keep on running?",
            "So what was Gandhi's work called?",
            "Which is the meaning of life?",
            "It is the desert. So what is life even?",
            "What Gaurav said.",
            "What Barack Obama ate this week?"
        ]
        return pos_ex, neg_ex


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    rule = RuleFour(nlp, verbose=True)

    rule.test()
