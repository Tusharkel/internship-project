"""
Singleton loader — loads model & vectorizer once at startup,
reuses across all requests.
"""

import joblib
from django.conf import settings


class ModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if not self._loaded:
            self.model      = joblib.load(settings.MODEL_PATH)
            self.vectorizer = joblib.load(settings.VECTORIZER_PATH)
            self._loaded    = True
            print("✅  ML model loaded into memory")

    def predict(self, text: str) -> dict:
        """Returns {'label': 'spam'|'ham', 'confidence': float}"""
        vec   = self.vectorizer.transform([text])
        label = self.model.predict(vec)[0]
        proba = self.model.predict_proba(vec)[0]
        confidence = float(max(proba))
        return {"label": label, "confidence": confidence}


# Global singleton instance
model_loader = ModelLoader()