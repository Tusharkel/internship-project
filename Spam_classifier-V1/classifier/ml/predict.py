from classifier.ml.preprocess import clean_text
from classifier.utils.model_loader import model_loader


def classify_email(subject: str, body: str) -> dict:
    """
    Combines subject + body, cleans text, returns prediction.
    Returns: { label: str, confidence: float }
    """
    raw_text = f"{subject} {body}".strip()
    clean    = clean_text(raw_text)
    return model_loader.predict(clean)