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
    # If last blank, say no.
    i = token.i
    l = len(token)

    if i == len(doc)-1:
        return False

    relevant_span = doc[i: i + 1]
    return relevant_span.text[l: l+1] == ' '
