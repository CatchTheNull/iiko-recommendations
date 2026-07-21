from django.urls import include, path, re_path
from django.views.generic import TemplateView
from django.views.static import serve

from config.settings import FRONTEND_DIST

urlpatterns = [
    path("api/", include("recommendations.urls")),
]

# Отдача собранного Vue-фронтенда (frontend/dist), если он есть.
if FRONTEND_DIST.exists():
    urlpatterns += [
        re_path(
            r"^assets/(?P<path>.*)$",
            serve,
            {"document_root": FRONTEND_DIST / "assets"},
        ),
        re_path(r"^(?!api/).*$", TemplateView.as_view(template_name="index.html")),
    ]
