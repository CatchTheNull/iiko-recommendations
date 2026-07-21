"""Клиенты внешних API: iiko Server (OLAP) и iikoTransport."""
from __future__ import annotations

import hashlib
import os

import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ─── iiko Server ─────────────────────────────────────────────────────────────

def auth_iiko(server_url: str, login: str, password: str) -> str:
    """Аутентификация на iiko Server. Возвращает ключ сессии."""
    sha1_pass = hashlib.sha1(password.encode()).hexdigest()
    url = f"{server_url.rstrip('/')}/resto/api/auth"
    resp = requests.get(url, params={"login": login, "pass": sha1_pass}, verify=False, timeout=30)
    resp.raise_for_status()
    key = resp.text.strip()
    if not key:
        raise ValueError("Сервер вернул пустой ключ авторизации")
    return key


def fetch_olap(server_url: str, key: str, date_from: str, date_to: str) -> pd.DataFrame:
    """Загружает данные о продажах через OLAP API iiko и возвращает DataFrame."""
    url = f"{server_url.rstrip('/')}/resto/api/v2/reports/olap"
    body = {
        "reportType": "SALES",
        "buildSummary": "false",
        "groupByRowFields": [
            "UniqOrderId.Id",
            "DishId",
            "DishName",
            "DishCategory",
        ],
        "filters": {
            "OpenDate.Typed": {
                "filterType": "DateRange",
                "periodType": "CUSTOM",
                "from": f"{date_from}T00:00:00.000",
                "to": f"{date_to}T00:00:00.000",
                "includeLow": True,
                "includeHigh": False,
            },
            "DeletedWithWriteoff": {
                "filterType": "IncludeValues",
                "values": ["NOT_DELETED"],
            },
            "OrderDeleted": {
                "filterType": "IncludeValues",
                "values": ["NOT_DELETED"],
            },
        },
    }
    # Ключ передаём и в query param и в cookie — как в рабочем curl
    resp = requests.post(
        url, params={"key": key}, cookies={"key": key},
        json=body, verify=False, timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame()

    # Сервер может вернуть либо список объектов {"field": value, ...}
    # либо {"columnNames": [...], "data": [[...], [...]]}
    if isinstance(rows[0], dict):
        df = pd.DataFrame(rows)
    else:
        columns = data.get("columnNames", [])
        df = pd.DataFrame(rows, columns=columns)

    # Нормализуем имена колонок к тому, что ожидает build_recommendations
    rename_map: dict[str, str] = {}
    if "DishCategory" in df.columns and "DishGroup.SecondParent" not in df.columns:
        rename_map["DishCategory"] = "DishGroup.SecondParent"
    if "DishId" not in df.columns and "DishName" in df.columns:
        df["DishId"] = df["DishName"]
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ─── iikoTransport API ───────────────────────────────────────────────────────

IIKO_TRANSPORT_BASE = "https://api-ru.iiko.services"

# Константы приложения для v2-авторизации (одинаковы для всех запросов).
# Можно переопределить переменными окружения IIKO_APP_ID и IIKO_CLIENT_SECRET.
IIKO_TRANSPORT_APP_ID = os.environ.get(
    "IIKO_APP_ID", "0490666c-0a38-4744-b2f7-e81ed047c366"
)
IIKO_TRANSPORT_CLIENT_SECRET = os.environ.get(
    "IIKO_CLIENT_SECRET", "P0I4gYuedJMCtdlcZQPRllpNnSzjwfgN7MM8hFSIK4g="
)


def iiko_transport_token(api_key: str) -> str:
    """POST /api/v2/access_token — получение Bearer-токена iikoTransport."""
    url = f"{IIKO_TRANSPORT_BASE}/api/v2/access_token"
    body = {
        "apiKey": api_key,
        "appId": IIKO_TRANSPORT_APP_ID,
        "clientSecret": IIKO_TRANSPORT_CLIENT_SECRET,
    }
    resp = requests.post(url, json=body, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # v2 может отдавать token / access_token / accessToken — берём что есть.
    token = (
        data.get("token")
        or data.get("access_token")
        or data.get("accessToken")
        or ""
    )
    if not token:
        raise ValueError(f"iikoTransport вернул пустой токен. Ответ: {str(data)[:300]}")
    return token


def iiko_transport_organizations(token: str) -> list[dict]:
    """GET /api/1/organizations — список организаций."""
    url = f"{IIKO_TRANSPORT_BASE}/api/1/organizations"
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("organizations", [])


def iiko_transport_menus(token: str, organization_id: str) -> list[dict]:
    """POST /api/2/menu — список внешних меню для организации."""
    url = f"{IIKO_TRANSPORT_BASE}/api/2/menu"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"organizationId": organization_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("externalMenus", [])


def iiko_transport_menu_items(
    token: str, external_menu_id: str, organization_id: str
) -> dict:
    """POST /api/2/menu/by_id — загружает внешнее меню, возвращает сырой JSON."""
    url = f"{IIKO_TRANSPORT_BASE}/api/2/menu/by_id"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={
            "externalMenuId": external_menu_id,
            "organizationIds": [organization_id],
            "asyncMode": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()
