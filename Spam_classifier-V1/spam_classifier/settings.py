import os
from pathlib import Path
from dotenv import load_dotenv
import mongoengine

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "fallback-secret")
DEBUG = os.getenv("DJANGO_DEBUG", "True") == "True"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.staticfiles",
    "rest_framework",
    "classifier",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "spam_classifier.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {"context_processors": []},
    }
]

WSGI_APPLICATION = "spam_classifier.wsgi.application"

# ── MongoDB via MongoEngine ──────────────────────────────────────────
mongoengine.connect(
    db=os.getenv("MONGO_DB_NAME", "spam_classifier_db"),
    host=os.getenv("MONGO_HOST", "localhost"),
    port=int(os.getenv("MONGO_PORT", 27017)),
)

# Disable Django's default ORM (no SQL DB needed)
DATABASES = {}

STATIC_URL = "/static/"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
}

MODEL_PATH = BASE_DIR / "models" / "naive_bayes_spam.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"