import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

__all__ = [
    "stop_words",
    "lemmatizer",
    "get_wordnet_pos",
    "cleaning_text",
]


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag: str):
    """Map POS tag to WordNet POS tag."""
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def cleaning_text(text):
    """
    End-to-end text cleaning pipeline:
    - lowercase
    - keep only alphabetic characters
    - tokenize
    - POS-tag-aware lemmatization
    - remove stopwords and very short tokens
    Returns a list of cleaned tokens.
    """
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    tokens = text.split()
    pos_tags = nltk.pos_tag(tokens)

    clean_tokens = []
    for word, tag in pos_tags:
        if word not in stop_words and len(word) > 2:
            wn_tag = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos=wn_tag)
            clean_tokens.append(lemma)

    return clean_tokens

