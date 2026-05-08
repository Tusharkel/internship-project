import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download once on first run
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Strip HTML tags
      3. Remove URLs
      4. Remove punctuation & digits
      5. Remove stopwords
      6. Stem tokens
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # strip URLs
    text = re.sub(r"\d+", " ", text)                 # remove digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stop_words and len(t) > 2]
    return " ".join(tokens)