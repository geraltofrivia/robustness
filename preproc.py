"""
    Code to preprocess given files and do basic NLP stuff (POS, NER etc)
"""
import spacy


class PreProcess:

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
            Class for pre-processing English sequences.

            **What happens inside&**:
                - spacy 2.0 pre-processing (default)

            :param spacy_model: str of the model name of spacy. Defaults to "en_core_web_sm"
        """
        self.nlp = spacy.load(spacy_model)

    def run(self, sequence: str):
        return self.nlp(sequence)
