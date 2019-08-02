import spacy


def need_space_after(token: spacy.tokens.Token) -> bool:
    """
        Finds whether a space occurs after this token in the doc or not.
        Eg.
            token= 'what'; sent= "what is the time"; op: True
            token= 'what'; sent= "what's the time"; op: False
    
    :param token: the token after which space may occur
    :return: bool
    """
    return len(token.text) != len(token.text_with_ws)


def need_space_after_(token: spacy.tokens.Token) -> str:
    """
        Uses need_space_after fn and returns a whitespace accordingly.

    :param token: the token after which space may occur
    :return: str
    """
    return ' ' if need_space_after(token) else ''