"""Простая сессионная авторизация: один логин/пароль из настроек."""
from __future__ import annotations

import hmac

from django.conf import settings
from rest_framework import serializers
from rest_framework.exceptions import NotAuthenticated, ValidationError
from rest_framework.response import Response
from rest_framework.views import APIView

SESSION_FLAG = "authenticated"


class ProtectedAPIView(APIView):
    """Базовый класс для эндпоинтов, требующих входа."""

    def initial(self, request, *args, **kwargs):
        super().initial(request, *args, **kwargs)
        if not request.session.get(SESSION_FLAG):
            raise NotAuthenticated("Требуется авторизация.")


class LoginSerializer(serializers.Serializer):
    login = serializers.CharField()
    password = serializers.CharField()


class LoginView(APIView):
    def post(self, request):
        s = LoginSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        login_ok = hmac.compare_digest(
            s.validated_data["login"].encode(), settings.APP_LOGIN.encode()
        )
        password_ok = hmac.compare_digest(
            s.validated_data["password"].encode(), settings.APP_PASSWORD.encode()
        )
        if not (login_ok and password_ok):
            raise ValidationError({"detail": "Неверный логин или пароль."})
        request.session[SESSION_FLAG] = True
        return Response({"authenticated": True})


class LogoutView(APIView):
    def post(self, request):
        request.session.flush()
        return Response({"authenticated": False})


class MeView(APIView):
    def get(self, request):
        return Response({"authenticated": bool(request.session.get(SESSION_FLAG))})
