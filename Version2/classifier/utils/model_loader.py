import joblib
import numpy as np
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
            self.model  = joblib.load(settings.MODEL_PATH)
            self.scaler = joblib.load(settings.VECTORIZER_PATH)
            self._loaded = True

    def predict(self, features: list) -> dict:
        """
        features: list of 3000 word-frequency floats
                  (matching Kaggle dataset columns)
        """
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        label     = self.model.predict(X_scaled)[0]
        proba     = self.model.predict_proba(X_scaled)[0]
        confidence = float(max(proba))
        return {"label": label, "confidence": round(confidence, 4)}


model_loader = ModelLoader()