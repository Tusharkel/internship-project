from django.apps import AppConfig


class ClassifierConfig(AppConfig):
    name = "classifier"

    def ready(self):
        # Load ML model once when Django starts
        from classifier.utils.model_loader import model_loader
        try:
            model_loader.load()
        except FileNotFoundError:
            print("⚠️  Model files not found. Run classifier/ml/train.py first.")