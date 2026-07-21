from django.urls import path

from . import auth, views

urlpatterns = [
    path("health", views.HealthView.as_view()),
    path("auth/login", auth.LoginView.as_view()),
    path("auth/logout", auth.LogoutView.as_view()),
    path("auth/me", auth.MeView.as_view()),
    path("transport/organizations", views.TransportOrganizationsView.as_view()),
    path("transport/menus", views.TransportMenusView.as_view()),
    path("recommendations", views.RecommendationsView.as_view()),
    path("recommendations/from-files", views.RecommendationsFromFilesView.as_view()),
]
