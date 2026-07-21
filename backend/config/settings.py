"""Настройки Django для сервера рекомендаций iiko."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# Собранный фронтенд (npm run build в frontend/) — отдаётся Django в проде.
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY", "django-insecure-dev-only-key-change-in-production"
)
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "rest_framework",
    "recommendations",
]

MIDDLEWARE = [
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
]

# Сессии в подписанных cookie — БД не нужна.
SESSION_ENGINE = "django.contrib.sessions.backends.signed_cookies"
SESSION_COOKIE_AGE = 14 * 24 * 3600  # 2 недели
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_HTTPONLY = True

# Учётные данные для входа в приложение (авторизация при первом входе).
# В проде обязательно переопредели через переменные окружения!
APP_LOGIN = os.environ.get("APP_LOGIN", "admin")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "admin")

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [FRONTEND_DIST],
        "APP_DIRS": False,
        "OPTIONS": {
            "context_processors": [],
            # Без кэша шаблонов: index.html меняется при каждой пересборке фронтенда.
            "loaders": ["django.template.loaders.filesystem.Loader"],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# БД не используется (сервис stateless), но Django требует конфигурацию.
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

LANGUAGE_CODE = "ru-ru"
TIME_ZONE = "UTC"
USE_TZ = True

STATIC_URL = "static/"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    # Публичный API без аутентификации/сессий.
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [],
    "UNAUTHENTICATED_USER": None,
    # Не перехватывать ?format= — мы используем его для выбора json/csv.
    "URL_FORMAT_OVERRIDE": None,
}

# Максимальный размер загружаемых файлов (XML выгрузки бывают большими)
DATA_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 100 * 1024 * 1024
