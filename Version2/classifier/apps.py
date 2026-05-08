from django.apps import AppConfig

class ClassifierConfig(AppConfig):
    name = "classifier"

    def ready(self):
        from classifier.utils.model_loader import model_loader
        try:
            model_loader.load()
            print("✅  ML model loaded successfully")
        except FileNotFoundError:
            print("⚠️   Model not found — run: python run_train.py")