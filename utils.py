import spacy


def need_space(token: spacy.tokens.Token, doc: spacy.tokens.Doc) -> bool:
    """
        Finds whether a space occurs after this token in the doc or not.
        Eg.
            token= 'what'; sent= "what is the time"; op: True
            token= 'what'; sent= "what's the time"; op: False
    
    :param token: the token after which space may occur
    :param doc: the doc/span/sent where the token exists
    :return: bool
    """
    return len(token.text) != len(token.text_with_ws)
