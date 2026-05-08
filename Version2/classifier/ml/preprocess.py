import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

_stemmer    = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"<[^>]+>",        " ", text)   # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # strip URLs
    text = re.sub(r"\d+",             " ", text)   # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [
        _stemmer.stem(t)
        for t in tokens
        if t not in _stop_words and len(t) > 2
    ]
    return " ".join(tokens)