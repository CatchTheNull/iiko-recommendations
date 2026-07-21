"""API рекомендаций «берут вместе» — замена Streamlit-приложения."""
from __future__ import annotations

import datetime
import io
import json

import pandas as pd
import requests
from django.http import HttpResponse
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .auth import ProtectedAPIView

from . import iiko_api
from .core import (
    build_recommendations,
    data_diagnostics,
    expand_allowed_ids,
    extract_external_menu_ids,
    parse_iiko_report_xml,
    parse_nomenclature_json,
)
from .serializers import (
    RecommendationsRequestSerializer,
    RecoSettingsSerializer,
    TransportAuthSerializer,
    TransportMenusSerializer,
)


class UpstreamError(APIException):
    status_code = 502
    default_detail = "Ошибка внешнего API"


def _raise_upstream(e: requests.exceptions.HTTPError):
    status = e.response.status_code if e.response is not None else "?"
    text = e.response.text[:300] if e.response is not None else str(e)
    raise UpstreamError(f"Ошибка внешнего API (HTTP {status}): {text}")


def _load_olap(cfg: dict) -> pd.DataFrame:
    try:
        key = iiko_api.auth_iiko(cfg["url"], cfg["login"], cfg["password"])
        return iiko_api.fetch_olap(cfg["url"], key, str(cfg["date_from"]), str(cfg["date_to"]))
    except requests.exceptions.HTTPError as e:
        _raise_upstream(e)
    except requests.exceptions.ConnectionError:
        raise UpstreamError("Не удалось подключиться к iiko Server. Проверь URL.")
    except ValueError as e:
        raise UpstreamError(str(e))


def _load_external_menu_ids(cfg: dict) -> set[str]:
    try:
        token = iiko_api.iiko_transport_token(cfg["api_key"])
        menu_data = iiko_api.iiko_transport_menu_items(
            token, cfg["external_menu_id"], cfg["organization_id"]
        )
    except requests.exceptions.HTTPError as e:
        _raise_upstream(e)
    except ValueError as e:
        raise UpstreamError(str(e))
    ext_ids, _ = extract_external_menu_ids(menu_data)
    if not ext_ids:
        raise ValidationError("Внешнее меню пустое: не найдено ни одной позиции.")
    return ext_ids


def _pipeline_response(df_raw: pd.DataFrame, settings: dict, allowed_ids: set[str] | None, fmt: str):
    diagnostics = data_diagnostics(df_raw, allowed_ids)

    wide_df, long_df = build_recommendations(
        df_raw.copy(),
        top_n=settings["top_n"],
        min_co=settings["min_co"],
        excluded_categories=set(settings["excluded_categories"]),
        allowed_ids=allowed_ids,
    )

    if fmt == "csv":
        buf = io.StringIO()
        long_df.to_csv(buf, index=False)
        resp = HttpResponse(buf.getvalue().encode("utf-8"), content_type="text/csv; charset=utf-8")
        filename = f"recommendations_{datetime.date.today()}.csv"
        resp["Content-Disposition"] = f'attachment; filename="{filename}"'
        return resp

    return Response({
        "diagnostics": diagnostics,
        "dishes": wide_df.to_dict(orient="records"),
        "recommendations": long_df.to_dict(orient="records"),
    })


class HealthView(APIView):
    def get(self, request):
        return Response({"status": "ok"})


class TransportOrganizationsView(ProtectedAPIView):
    """Список организаций iikoTransport (для выбора organization_id)."""

    def post(self, request):
        s = TransportAuthSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        try:
            token = iiko_api.iiko_transport_token(s.validated_data["api_key"].strip())
            orgs = iiko_api.iiko_transport_organizations(token)
        except requests.exceptions.HTTPError as e:
            _raise_upstream(e)
        except ValueError as e:
            raise UpstreamError(str(e))
        return Response({
            "organizations": [{"id": o.get("id"), "name": o.get("name")} for o in orgs]
        })


class TransportMenusView(ProtectedAPIView):
    """Список внешних меню организации (для выбора external_menu_id)."""

    def post(self, request):
        s = TransportMenusSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        try:
            token = iiko_api.iiko_transport_token(s.validated_data["api_key"].strip())
            menus = iiko_api.iiko_transport_menus(token, s.validated_data["organization_id"])
        except requests.exceptions.HTTPError as e:
            _raise_upstream(e)
        except ValueError as e:
            raise UpstreamError(str(e))
        return Response({
            "external_menus": [{"id": m.get("id"), "name": m.get("name")} for m in menus]
        })


class RecommendationsView(ProtectedAPIView):
    """Полный пайплайн: продажи из iiko Server (+ опционально фильтр по внешнему
    меню iikoTransport) → таблица рекомендаций. `?format=csv` — CSV-файл."""

    parser_classes = [JSONParser]

    def post(self, request):
        s = RecommendationsRequestSerializer(data=request.data)
        s.is_valid(raise_exception=True)
        data = s.validated_data

        df_raw = _load_olap(data["iiko_server"])
        if df_raw.empty:
            raise ValidationError("OLAP вернул 0 строк за выбранный период.")

        allowed_ids = None
        if data.get("transport"):
            allowed_ids = _load_external_menu_ids(data["transport"])

        fmt = request.query_params.get("format", "json")
        return _pipeline_response(df_raw, data["settings"], allowed_ids, fmt)


class RecommendationsFromFilesView(ProtectedAPIView):
    """Пайплайн из файлов — аналог вкладки «XML / JSON файлы» в старом UI.

    multipart/form-data:
      - sales_xml (обязательно): XML выгрузка iiko
      - menu_json (опц.): JSON внешнего меню (/api/2/menu/by_id)
      - nomenclature_json (опц.): JSON номенклатуры iiko Server
      - top_n, min_co, excluded_categories (по одной в строке)
    """

    parser_classes = [MultiPartParser]

    def post(self, request):
        sales_xml = request.FILES.get("sales_xml")
        if sales_xml is None:
            raise ValidationError("Файл sales_xml обязателен.")

        try:
            df_raw = parse_iiko_report_xml(sales_xml.read())
        except ValueError as e:
            raise ValidationError(str(e))
        except Exception:
            raise ValidationError("Не удалось прочитать XML файл продаж.")

        allowed_ids = None
        menu_json = request.FILES.get("menu_json")
        if menu_json is not None:
            try:
                menu_data = json.loads(menu_json.read())
            except Exception:
                raise ValidationError("Не удалось прочитать JSON внешнего меню.")
            allowed_ids, _ = extract_external_menu_ids(menu_data)

            nomenclature_json = request.FILES.get("nomenclature_json")
            if nomenclature_json is not None:
                try:
                    nom_data = json.loads(nomenclature_json.read())
                except Exception:
                    raise ValidationError("Не удалось прочитать JSON номенклатуры.")
                allowed_ids = expand_allowed_ids(allowed_ids, parse_nomenclature_json(nom_data))

        settings_serializer = RecoSettingsSerializer(data={
            "top_n": request.data.get("top_n", 8),
            "min_co": request.data.get("min_co", 1),
            "excluded_categories": [
                x.strip()
                for x in str(request.data.get("excluded_categories", "")).splitlines()
                if x.strip()
            ],
        })
        settings_serializer.is_valid(raise_exception=True)

        fmt = request.query_params.get("format", "json")
        return _pipeline_response(df_raw, settings_serializer.validated_data, allowed_ids, fmt)
