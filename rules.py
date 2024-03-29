""" File which has implementation of all the rules"""
import random
from typing import Union
from sklearn.metrics import confusion_matrix

# spaCy imports
import spacy
from spacy.matcher import Matcher

# pattern imports
from pattern.en import conjugate, lemma, lexeme, tenses

# Local imports
import utils
from resources import resources as res, gb_us

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

    def test(self, prob=1):
        """ Go through pos and neg examples, and find what changed and what didn't. """
        print(f" -- Testing {type(self)} --")

        pos_ex, neg_ex = self.examples()
        y_true = [1 for _ in pos_ex] + [0 for _ in neg_ex]
        y_pred = []

        # Document-ize these
        pos_ex = [self.nlp(doc) for doc in pos_ex]
        neg_ex = [self.nlp(doc) for doc in neg_ex]

        for doc in pos_ex + neg_ex:
            changed, op = self.apply(doc, prob)
            y_pred.append(1 if changed and op.text != doc.text else 0)

            if self.verbose:
                print("BEFORE:", doc)
                print("AFTER :", op)
                print('---')

        print(y_true)
        print(y_pred)
        print(confusion_matrix(y_true, y_pred))

    @staticmethod
    def probabilistic_filter(prob):
        """Syntactic sugar. Whether a rule should be applied or not."""
        return random.uniform(0,1) < prob

class Rule1(Rule):
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
            if self._can_be_applied_(sentence) and self.probabilistic_filter(probability):
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


class Rule2(Rule):
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
        self.matcher.add("Rule2", None, self.pattern)

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> Union[str, spacy.tokens.doc.Doc]:
        """See base class"""
        applied = False
        matches = self.matcher(sequence)
        alt_sequence = ''

        old_end_id = 0
        for match_id, start_id, end_id in matches:
            if not self.probabilistic_filter(probability):
                continue
            applied = True
            if start_id == 0:
                alt_sequence += "Which "
                old_end_id = end_id - 1
            else:
                alt_sequence += sequence[old_end_id:start_id].text
                alt_sequence += " Which "
                old_end_id = end_id - 1
        alt_sequence += sequence[old_end_id:].text

        return (True, self.nlp(alt_sequence)) if applied else (False, sequence)

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


class Rule3(Rule):
    """
        **Transformation**:
            What VERB -> So what VERB

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rules for Debugging NLP Models

        **Examples**
            before: What was Gandhi's work called?
            after: So what was Gandhi's work called?

        **Comment**
            Original intended task was `Machine Comprehension/Visual Question Answering`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pattern = [{'LOWER': 'what'}, {'POS': 'VERB'}]
        self.matcher.add("Rule2", None, self.pattern)

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
            if not self.probabilistic_filter(probability):
                continue
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


class Rule4(Rule):
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
            if sentence[0].lower_ == 'what' and sentence[1].tag_ == 'VBD' and self.probabilistic_filter(probability):
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


class Rule5(Rule):
    """
        **Transformation**:
            WP VBZ -> WP's

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rulesfor Debugging NLP Models

        **Examples**
            before: What has been cut?
            after: What's been cut?

        **Comment**
        Original intended task was `Visual Question Answering`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pattern = [{'TAG': 'WP'}, {'TAG': 'VBZ'}]
        self.matcher.add("Rule2", None, self.pattern)

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> (bool, Union[str, spacy.tokens.doc.Doc]):
        """See base class"""
        applied = False
        matches = self.matcher(sequence)
        alt_sequence = ''

        old_end_id = 0
        for match_id, start_id, end_id in matches:
            if not self.probabilistic_filter(probability):
                continue

            applied = True
            if start_id == 0:
                alt_sequence += sequence[start_id].text + "'s"
                alt_sequence += utils.need_space_after_(token=sequence[end_id])
                old_end_id = end_id
            else:
                # Add existing sequence to the alt_seq
                alt_sequence += sequence[old_end_id:start_id].text_with_ws

                # If we're in a new sentence, add "So" else "so"
                alt_sequence += sequence[start_id].text + "'s"
                alt_sequence += utils.need_space_after_(token=sequence[end_id])
                old_end_id = end_id
        alt_sequence += sequence[old_end_id:].text

        return (True, self.nlp(alt_sequence)) if applied else (False, sequence)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            "What is being cut?",
            "What has been cut? ",
            "Here's what is: this shit sucks.",
            "Is this what is being cut?"
        ]
        neg_ex += [
            "What will be cut?",
            "Which running is preferred.",
            "Where is the crowd running towards?",
            "Is this what putting feels like?",
            "What Barack Obama ate this week?"
        ]
        return pos_ex, neg_ex


class Rule6(Rule):
    """
        **Transformation**:
            ..o.. -> ..ou.. (british-english conversion.)

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rulesfor Debugging NLP Models

        **Examples**
            before: color
            after: colour

        **Comment**
        Original intended task was `Visual Question Answering`
        Using word pairs from
            https://stackoverflow.com/questions/42329766/python-nlp-british-english-vs-american-english
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_pairs = gb_us.us_bg_lexpairs

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> (bool, Union[str, spacy.tokens.doc.Doc]):
        """ Simply lookup words and replace """
        alt_sequence = ''
        applied = False
        old_end_id = 0

        for tok in sequence:
            if tok.lower_ in self.word_pairs and not self.probabilistic_filter(probability):
                applied = True
                alt_sequence += sequence[old_end_id:tok.i].text_with_ws
                alt_sequence += self.word_pairs[tok.lower_]
                alt_sequence += utils.need_space_after_(tok)
                old_end_id = tok.i + 1

        alt_sequence += sequence[old_end_id:].text_with_ws

        return (applied, self.nlp(alt_sequence)) if applied else (applied, sequence)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            'What counselor Hahn wanted, only he knows.',
            'which color is phony, do you reckon?'
        ]
        neg_ex += [
            'this is the life we chose'
        ]
        return pos_ex, neg_ex


class Rule7(Rule):
    """
        **Transformation**:
            ..ou.. -> ..o.. (british-english conversion.)

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rulesfor Debugging NLP Models

        **Examples**
            before: colour
            after: color

        **Comment**
        Original intended task was `Visual Question Answering`
        Using word pairs from
            https://stackoverflow.com/questions/42329766/python-nlp-british-english-vs-american-english
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_pairs = {v:k for k,v in gb_us.us_bg_lexpairs.items()}

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> (bool, Union[str, spacy.tokens.doc.Doc]):
        """ Simply lookup words and replace """
        alt_sequence = ''
        applied = False
        old_end_id = 0

        for tok in sequence:
            if tok.lower_ in self.word_pairs:
                applied = True
                alt_sequence += sequence[old_end_id:tok.i].text_with_ws
                alt_sequence += self.word_pairs[tok.lower_]
                alt_sequence += utils.need_space_after_(tok)
                old_end_id = tok.i + 1

        alt_sequence += sequence[old_end_id:].text_with_ws

        return (applied, self.nlp(alt_sequence)) if applied else (applied, sequence)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            'What counsellor Hahn wanted, only he knows.',
            'which colour is phoney, do you reckon?'
        ]
        neg_ex += [
            'this is the life we chose'
        ]
        return pos_ex, neg_ex


class Rule8(Rule):
    """
        **Transformation**:
            ADV VBZ -> ADV's

        **Source**:
            [Paper] Semantically Equivalent Adversarial Rulesfor Debugging NLP Models

        **Examples**
            before: Where is the jet?
            after: Where's the jet?

        **Comment**
        Original intended task was `Visual Question Answering`
        If the VBZ word has a "'s" or similar, skip this one.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pattern = [{'POS': 'ADV'}, {'TAG': 'VBZ'}]
        self.matcher.add("Rule2", None, self.pattern)

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> (bool, Union[str, spacy.tokens.doc.Doc]):
        """See base class"""
        applied = False
        matches = self.matcher(sequence)
        alt_sequence = ''

        old_end_id = 0
        for match_id, start_id, end_id in matches:
            if sequence[end_id].text[0] == "'":
                # The token is a 's or 'd, skip
                continue

            applied = True
            alt_sequence += sequence[old_end_id: start_id].text_with_ws
            alt_sequence += sequence[start_id].text
            alt_sequence += "'s"
            alt_sequence += utils.need_space_after_(sequence[end_id])
            old_end_id = end_id

        alt_sequence += sequence[old_end_id:].text_with_ws

        if applied:
            return True, self.nlp(alt_sequence)
        else:
            return False, sequence

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            "Where is the jet?",
            "How is the jet?",
            "And how is the jet?",
            "Would you be so kind as to illustrate exactly how is the potato?",
            "Where is the jet?"
        ]
        neg_ex += [
            "How was the tomato?",
            "Where's the jet coming from?",
            "Where'd the jet come from?",
            "Where did the jet come from?"
        ]

        return pos_ex, neg_ex


class Rule13(Rule):
    """
        **Transformation**:
            this -> that

        **Source**:
            [Paper] The CoNLL-2014 Shared Task on Grammatical Error Correction

        **Examples**
            before: Now this is a movie I like
            after: Now that is a movie I like

        **Comment**
            Original intended task was `Grammar correction`
        """
    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1) -> (bool, Union[str, spacy.tokens.doc.Doc]):
        """ see base class """
        applied = False
        old_end_id = 0
        alt_sequence = ''

        for token in sequence:
            if token.lower_ == 'this':
                applied = True
                alt_sequence += sequence[old_end_id:token.i].text_with_ws
                alt_sequence += 'that'
                alt_sequence += utils.need_space_after_(token)

                old_end_id = token.i + 1

        alt_sequence += sequence[old_end_id:].text_with_ws

        if applied:
            return applied, self.nlp(alt_sequence)
        else:
            return applied, sequence

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            "Is this the real life?",
            "tell me this: who told you this?",
            "tell me this: who told you that?",
            "this that this that and then some more of this.",
            "What if the last word was this",
            "and what if it were 'this'",
            "this's how its done!"
        ]
        neg_ex += [
            "That's how its done",
            "tell me that: who told you that",
            "the rose and the thistle they both have thorns.",
            "Where did the jet come from?"
        ]

        return pos_ex, neg_ex


class Rule15(Rule):
    """
        **Transformation**:

            To create inflexions by switching one of the verb in the sentence to its other tense. This would result in
            verb disagreement.

        **Source**:
            [Paper] The CoNLL-2014 Shared Task on Grammatical Error Correction

        **Examples**
            before: Medical technology during that time *was* not advanced enough to cure him. ->
            after: Medical technology during that time *is* not advanced enough to cure him.

        **Comment**
            Original intended task was `Grammar correction`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pattern = [{'POS': 'VERB'}]
        self.matcher.add("VerbSwitch", None, self.pattern)

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1.0) -> (
    bool, Union[str, spacy.tokens.doc.Doc]):
        """See base class"""
        applied = False
        alt_sequence = ''

        '''
            Rule application logic
                -> if single 'VERB' appears in seq
                    -> Flip it with some randomly selected variant of its verb variant 
                -> if more than one verb appears in the sequence, 
                    -> Select one of the verb randomly and flip it. 
        '''

        for sent in sequence.sents:

            # Apply matcher at each instance.
            matches = self.matcher(sent.as_doc())  # as_doc might be buggy.
            seq = ''

            # Randomly select one if the number oif matches are less than 3 else select 2 (arbitary number)
            if matches:
                if len(matches) < 3:
                    match = random.choices(matches, k=1)
                else:
                    match = random.choices(matches, k=1)

            else:
                alt_sequence += sent[:].text_with_ws
                continue

            old_end_id = 0

            for match_id, start_id, end_id in match:

                # Generate the word.
                new_word = self.verb_fom(word=sent[start_id].text)

                # if the new generated word is not same than we have inflected the sentence.
                if new_word != sent[start_id].text:
                    applied = True

                if start_id != 0:
                    seq += sent[old_end_id:start_id].text_with_ws
                    seq += new_word
                    seq += utils.need_space_after_(token=sequence[start_id])
                    old_end_id = end_id

                else:
                    seq += new_word.capitalize()  # Capitalize it.
                    seq += utils.need_space_after_(token=sequence[start_id])
                    old_end_id = end_id

            seq += sent[old_end_id:].text_with_ws
            alt_sequence += seq
        return (True, self.nlp(alt_sequence)) if applied else (False, sequence)

    @staticmethod
    def examples():
        pos_ex, neg_ex = [], []
        pos_ex += [
            "Medical technology during that time was not advanced enough to cure him. This is America",
            "Medical technology during that time is not advanced enough to cure him.",
            "This is America!"
        ]
        return pos_ex, neg_ex

    @staticmethod
    def verb_fom(word: str) -> str:

        # Step 1: check if the word is in present tense or past.
        tense_list = tenses(word)

        if tense_list is None or len(tense_list) == 0:
            return word

        present_tense = True

        for index, i in enumerate(tense_list):
            if i[0] == 'present':
                present_tense = True
                final_tense = i
                break
            if i[0] == 'past':
                present_tense = False
                final_tense = i
                break

        if present_tense:
            tense_string = "past"
        else:
            tense_string = "present"

        # Step 2: Create
        new_word = conjugate(word, tense=tense_string, person=final_tense[1], number=final_tense[2], negated=False)

        # Step 3: Return the word
        return new_word if new_word is not None else word


class Rule16(Rule):
    """
        **Transformation**:

            To create inflexions by switching one of the modal verb in the sentence to its other tense.
            This would result in modal verb disagreement.

        **Source**:
            [Paper] The CoNLL-2014 Shared Task on Grammatical Error Correction

        **Examples**
            before: Although the problem would not be serious, people would still be afraid.
            after: Although the problem may not be serious, people might still be afraid.

        **Comment**
            Original intended task was `Grammar correction`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for mod in res.modal_verbs:
            self.matcher.add(mod,None,[{"LOWER":mod}])

    def apply(self, sequence: spacy.tokens.doc.Doc, probability: float = 1.0) -> (
    bool, Union[str, spacy.tokens.doc.Doc]):
        """See base class"""
        applied = False
        alt_sequence = ''

        '''
            Rule application logic
                -> if single 'VERB' appears in seq
                    -> Flip it with some randomly selected variant of its verb variant 
                -> if more than one verb appears in the sequence, 
                    -> Select one of the verb randomly and flip it. 
        '''

        for sent in sequence.sents:

            # Apply matcher at each instance.
            matches = self.matcher(sent.as_doc())  # as_doc might be buggy.
            seq = ''

            # Randomly select one if the number oif matches are less than 3 else select 2 (arbitary number)
            if matches:
                if len(matches) < 3:
                    match = random.choices(matches, k=1)
                else:
                    match = random.choices(matches, k=1)

            else:
                alt_sequence += sent[:].text_with_ws
                continue

            old_end_id = 0

            for match_id, start_id, end_id in match:

                # Generate the word.
                new_word = self.verb_fom(word=sent[start_id].text)

                # if the new generated word is not same than we have inflected the sentence.
                if new_word != sent[start_id].text:
                    applied = True

                if start_id != 0:
                    seq += sent[old_end_id:start_id].text_with_ws
                    seq += new_word
                    seq += utils.need_space_after_(token=sequence[start_id])
                    old_end_id = end_id

                else:
                    seq += new_word.capitalize()  # Capitalize it.
                    seq += utils.need_space_after_(token=sequence[start_id])
                    old_end_id = end_id

            seq += sent[old_end_id:].text_with_ws
            alt_sequence += seq
        return (True, self.nlp(alt_sequence)) if applied else (False, sequence)

    @staticmethod
    def examples():
        # @TODO: Add more examples
        pos_ex, neg_ex = [], []
        pos_ex += [
            "Although the problem would not be serious, people would still be afraid",
        ]
        neg_ex += [
            "Medical technology during that time is not advanced enough to cure him.",
            "This is America!"
        ]
        return pos_ex, neg_ex

    @staticmethod
    def verb_fom(word: str) -> str:

        _k = 5
        while _k != 0:
            new_word = random.choices(res.modal_verbs)[0]
            if new_word != word:
                return new_word
            else:
                _k = _k - 1

        return word


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")
    rule = Rule1(nlp, verbose=True)

    rule.test(prob=0.3)
