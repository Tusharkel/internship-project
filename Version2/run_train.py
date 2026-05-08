import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "spam_classifier.settings")

from classifier.ml.train import train

if __name__ == "__main__":
    train()