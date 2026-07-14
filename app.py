from __future__ import annotations

import datetime
import hashlib
import io
import itertools
import json
import os
import urllib3
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
import requests
import streamlit as st

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REQUIRED_FIELDS = [
    "UniqOrderId.Id",
    "DishId",
    "DishName",
    "DishGroup.SecondParent",
]


def _text(x):
    if x is None:
        return None
    t = (x.text or "").strip()
    return t if t else None


def parse_iiko_report_xml(xml_bytes: bytes) -> pd.DataFrame:
    """
    Best-effort parser for iiko XML exports.
    Tries to extract rows that contain the required fields:
    UniqOrderId.Id, DishId, DishName, DishGroup.SecondParent
    """
    root = ET.fromstring(xml_bytes)

    # Strategy A: elements that have child-tags equal to required fields
    rows = []
    for el in root.iter():
        # Fast check: does this element contain at least 2 children?
        children = list(el)
        if len(children) < 2:
            continue

        child_tags = {c.tag for c in children}
        if any(f in child_tags for f in REQUIRED_FIELDS):
            row = {}
            for c in children:
                if c.tag in REQUIRED_FIELDS:
                    row[c.tag] = _text(c)
            # Some iiko exports use nested tags or different naming; keep only if essentials exist
            if row.get("UniqOrderId.Id") and row.get("DishId") and row.get("DishName"):
                rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        return df

    # Strategy B: "field/value" style (e.g., <field name="DishId">..</field>)
    rows = []
    for el in root.iter():
        # Look for nodes that contain <field name="...">value</field> pattern
        fields = {}
        for c in list(el):
            name = c.attrib.get("name") or c.attrib.get("Name") or c.attrib.get("field") or c.attrib.get("Field")
            if not name:
                continue
            if name in REQUIRED_FIELDS:
                fields[name] = _text(c) or c.attrib.get("value") or c.attrib.get("Value")
        if fields.get("UniqOrderId.Id") and fields.get("DishId") and fields.get("DishName"):
            rows.append(fields)

    if rows:
        return pd.DataFrame(rows)

    # If nothing worked, raise a readable error
    raise ValueError(
        "Не смог распарсить XML: не нашёл строки с полями UniqOrderId.Id, DishId, DishName. "
        "Возможен нестандартный формат выгрузки (например Spreadsheet XML)."
    )


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
    # DishCategory → DishGroup.SecondParent
    if "DishCategory" in df.columns and "DishGroup.SecondParent" not in df.columns:
        rename_map["DishCategory"] = "DishGroup.SecondParent"
    # Если DishId отсутствует — используем DishName как идентификатор
    if "DishId" not in df.columns and "DishName" in df.columns:
        df["DishId"] = df["DishName"]
    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# ─── iikoTransport API ───────────────────────────────────────────────────────

IIKO_TRANSPORT_BASE = "https://api-ru.iiko.services"

# Константы приложения для v2-авторизации (одинаковы для всех запросов).
# По умолчанию берутся отсюда; при желании их можно спрятать в переменные
# окружения / HF Secrets IIKO_APP_ID и IIKO_CLIENT_SECRET.
IIKO_TRANSPORT_APP_ID = os.environ.get(
    "IIKO_APP_ID", "0490666c-0a38-4744-b2f7-e81ed047c366"
)
IIKO_TRANSPORT_CLIENT_SECRET = os.environ.get(
    "IIKO_CLIENT_SECRET", "P0I4gYuedJMCtdlcZQPRllpNnSzjwfgN7MM8hFSIK4g="
)


def iiko_transport_token(api_key: str) -> str:
    """POST /api/v2/access_token — получение Bearer-токена iikoTransport.

    В форме вводится apiKey; appId и clientSecret — константы приложения.
    """
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


def extract_external_menu_ids(menu_data: dict) -> tuple[set[str], dict[str, str]]:
    """Извлекает из JSON внешнего меню множество всех ID для матчинга.

    Собирает И itemId (UUID) И sku — чтобы матч работал независимо от того,
    что OLAP возвращает как DishId (UUID или артикул).

    Returns:
        (all_ids, id_to_name): множество всех ID + маппинг id→name для отображения.
    """
    all_ids: set[str] = set()
    id_to_name: dict[str, str] = {}

    for category in menu_data.get("itemCategories", []):
        for item in category.get("items", []):
            name = item.get("name", "")
            # itemId — UUID, совпадает с id в номенклатуре и DishId из OLAP
            item_id = item.get("itemId", "")
            if item_id:
                all_ids.add(str(item_id))
                id_to_name[str(item_id)] = name
            # sku — артикул (num), на случай если DishId = артикул
            sku = item.get("sku", "")
            if sku:
                all_ids.add(str(sku))
                id_to_name[str(sku)] = name

    return all_ids, id_to_name


def parse_nomenclature_json(data: list | dict) -> dict[str, str]:
    """Парсит JSON номенклатуры iiko Server.

    Строит маппинг id (UUID) → num (артикул) для конвертации ID.
    Номенклатура может быть списком или dict с ключом 'products'.
    """
    products = data if isinstance(data, list) else data.get("products", [])
    id_to_num: dict[str, str] = {}
    for p in products:
        pid = str(p.get("id", ""))
        num = str(p.get("num", ""))
        if pid and num:
            id_to_num[pid] = num
    return id_to_num


def build_recommendations(
    df: pd.DataFrame,
    top_n: int = 8,
    min_co: int = 1,
    excluded_categories: set[str] | None = None,
    allowed_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Строит рекомендации «берут вместе» из данных продаж.

    Args:
        df: DataFrame с колонками REQUIRED_FIELDS
        top_n: макс. рекомендаций на блюдо
        min_co: мин. число совместных покупок
        excluded_categories: категории для исключения
        allowed_ids: если задано — оставить только блюда с этими dish_id

    Returns:
      - wide_df: one row per dish with recommendations list string
      - long_df: one row per (dish -> recommended dish) with rank & co_occurrence (for CSV)
    """
    excluded_categories = excluded_categories or set()

    # Normalize columns
    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            df[col] = None

    df = df.rename(columns={
        "UniqOrderId.Id": "order_id",
        "DishId": "dish_id",
        "DishName": "dish_name",
        "DishGroup.SecondParent": "category",
    }).copy()

    # Basic cleanup
    df["order_id"] = df["order_id"].astype(str)
    df["dish_id"] = df["dish_id"].astype(str)
    df["dish_name"] = df["dish_name"].astype(str)
    df["category"] = df["category"].fillna("").astype(str)

    # Exclude categories (e.g., modifiers)
    if excluded_categories:
        df = df[~df["category"].isin(excluded_categories)].copy()

    # Фильтрация по внешнему меню — ДО построения co-occurrence
    if allowed_ids:
        df = df[df["dish_id"].isin(allowed_ids)].copy()

    # Build order -> unique dish_ids
    order_to_dishes: dict[str, set[str]] = defaultdict(set)
    dish_name_map: dict[str, str] = {}
    dish_cat_map: dict[str, str] = {}

    for r in df.itertuples(index=False):
        order_to_dishes[r.order_id].add(r.dish_id)
        # Keep last-seen name/category (good enough for exports)
        dish_name_map[r.dish_id] = r.dish_name
        dish_cat_map[r.dish_id] = r.category

    # Co-occurrence counts (undirected pairs)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for order_id, dishes in order_to_dishes.items():
        if len(dishes) < 2:
            continue
        for a, b in itertools.combinations(sorted(dishes), 2):
            pair_counts[(a, b)] += 1

    # Build directed recommendations: for each dish, list other dish with counts
    rec_map: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (a, b), cnt in pair_counts.items():
        if cnt < min_co:
            continue
        rec_map[a].append((b, cnt))
        rec_map[b].append((a, cnt))

    # Long table (one row per recommendation)
    long_rows = []
    for dish_id, recs in rec_map.items():
        # sort: by count desc, then by name asc for stability
        recs_sorted = sorted(
            recs,
            key=lambda x: (-x[1], (dish_name_map.get(x[0], "") or "").lower(), x[0])
        )[:top_n]

        for rank, (rec_id, cnt) in enumerate(recs_sorted, start=1):
            long_rows.append({
                "dish_id": dish_id,
                "dish_name": dish_name_map.get(dish_id, ""),
                "category": dish_cat_map.get(dish_id, ""),
                "recommended_dish_id": rec_id,
                "recommended_dish_name": dish_name_map.get(rec_id, ""),
                "rank": rank,
                "co_occurrence": cnt,
            })

    long_df = pd.DataFrame(long_rows)

    # Remove dishes with no recs (already excluded by construction, but keep safe)
    if long_df.empty:
        wide_df = pd.DataFrame(columns=["dish_id", "dish_name", "category", "recommendations"])
        return wide_df, long_df

    # Wide table: group recs into a single string list for UI
    def _format_recs(g: pd.DataFrame) -> str:
        # "Name (ID)" items, ordered by rank
        g2 = g.sort_values("rank")
        items = [f"{row.recommended_dish_name} ({row.recommended_dish_id})" for row in g2.itertuples()]
        return " | ".join(items)

    wide_df = (long_df
               .groupby(["dish_id", "dish_name", "category"], as_index=False)
               .apply(lambda g: pd.Series({"recommendations": _format_recs(g)}))
               .reset_index(drop=True))

    # Sort wide by number of recs then name
    wide_df["rec_count"] = wide_df["recommendations"].apply(lambda s: 0 if not s else s.count("|") + 1)
    wide_df = wide_df.sort_values(["rec_count", "dish_name"], ascending=[False, True]).drop(columns=["rec_count"])

    return wide_df, long_df


# ─── UI ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="iiko Recommendations", layout="wide")
st.title("iiko: рекомендации «берут вместе»")

with st.sidebar:
    st.header("Настройки")
    top_n = st.slider("Макс. рекомендаций на блюдо", min_value=1, max_value=8, value=8, step=1)
    min_co = st.number_input("Мин. совместных покупок (co-occurrence)", min_value=1, value=1, step=1)

    st.caption("Исключения (обычно модификаторы):")
    excluded_text = st.text_area("Категории для исключения (по одной в строке)", value="")
    excluded_categories = {x.strip() for x in excluded_text.splitlines() if x.strip()}

# ─── Вкладки загрузки данных ──────────────────────────────────────────────────

tab_combined, tab_api, tab_xml = st.tabs([
    "🚀 iiko Server + iikoTransport",
    "🔌 iiko Server (только продажи)",
    "📂 XML / JSON файлы",
])

# ── Вкладка 1: Комбинированная — iiko Server + iikoTransport ──

with tab_combined:
    st.subheader("Подключение к iiko Server + iikoTransport")
    st.caption(
        "Загрузи данные продаж с iiko Server и внешнее меню из iikoTransport. "
        "Рекомендации будут содержать только позиции из внешнего меню."
    )

    st.markdown("---")
    st.markdown("#### 1. Данные продаж (iiko Server)")

    server_url = st.text_input(
        "URL сервера",
        placeholder="https://hostname:443",
        key="c_server_url",
    )

    col_l, col_p = st.columns(2)
    with col_l:
        login = st.text_input("Логин", key="c_login")
    with col_p:
        password = st.text_input("Пароль", type="password", key="c_password")

    col_f, col_t = st.columns(2)
    with col_f:
        date_from = st.date_input(
            "Дата с",
            value=datetime.date.today() - datetime.timedelta(days=30),
            key="c_date_from",
        )
    with col_t:
        date_to = st.date_input(
            "Дата по",
            value=datetime.date.today(),
            key="c_date_to",
        )

    if st.button("Загрузить данные с сервера", key="c_btn_load_olap"):
        if not server_url or not login or not password:
            st.error("Заполни URL сервера, логин и пароль.")
        else:
            try:
                with st.spinner("Аутентификация..."):
                    key = auth_iiko(server_url, login, password)
                with st.spinner(f"Загрузка данных за {date_from} – {date_to}..."):
                    df_loaded = fetch_olap(server_url, key, str(date_from), str(date_to))
                st.session_state["df_raw"] = df_loaded
                st.success(f"Получено строк: {len(df_loaded):,}")
            except requests.exceptions.HTTPError as e:
                st.error(f"Ошибка HTTP {e.response.status_code}: {e.response.text[:300]}")
            except requests.exceptions.ConnectionError:
                st.error("Не удалось подключиться к серверу. Проверь URL.")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.markdown("#### 2. Внешнее меню (iikoTransport)")

    api_login = st.text_input(
        "API-ключ iikoTransport (apiKey)",
        type="password",
        key="c_transport_api_login",
        placeholder="37ac8a1d7eb9446281c4934c0ba8f3f3",
    )

    if st.button("Получить токен и организации", key="c_btn_transport_token"):
        if not api_login.strip():
            st.error("Укажи API-ключ iikoTransport.")
        else:
            try:
                with st.spinner("Получение токена..."):
                    tok = iiko_transport_token(api_login.strip())
                st.session_state["transport_token"] = tok
                with st.spinner("Загрузка организаций..."):
                    orgs = iiko_transport_organizations(tok)
                st.session_state["transport_orgs"] = orgs
                for k in ["transport_menus", "external_menu_ids", "external_menu_names"]:
                    st.session_state.pop(k, None)
                st.success(f"Токен получен. Организаций: {len(orgs)}")
            except requests.exceptions.HTTPError as e:
                st.error(f"Ошибка HTTP {e.response.status_code}: {e.response.text[:300]}")
            except Exception as e:
                st.error(str(e))

    # Выбор организации
    if "transport_orgs" in st.session_state and st.session_state["transport_orgs"]:
        orgs = st.session_state["transport_orgs"]
        org_options = {f"{o['name']} ({o['id'][:8]}…)": o["id"] for o in orgs}
        selected_org_label = st.selectbox(
            "Организация",
            options=list(org_options.keys()),
            key="c_transport_org_select",
        )
        selected_org_id = org_options[selected_org_label]

        if st.button("Загрузить внешние меню", key="c_btn_transport_menus"):
            try:
                tok = st.session_state["transport_token"]
                with st.spinner("Загрузка списка меню..."):
                    menus = iiko_transport_menus(tok, selected_org_id)
                st.session_state["transport_menus"] = menus
                st.session_state["transport_selected_org_id"] = selected_org_id
                for k in ["external_menu_ids", "external_menu_names"]:
                    st.session_state.pop(k, None)
                if menus:
                    st.success(f"Найдено внешних меню: {len(menus)}")
                else:
                    st.warning("Внешних меню не найдено для этой организации.")
            except Exception as e:
                st.error(str(e))

    # Выбор внешнего меню и загрузка
    if "transport_menus" in st.session_state and st.session_state["transport_menus"]:
        menus = st.session_state["transport_menus"]
        menu_options = {f"{m['name']} (id: {m['id']})": m["id"] for m in menus}
        selected_menu_label = st.selectbox(
            "Внешнее меню",
            options=list(menu_options.keys()),
            key="c_transport_menu_select",
        )
        selected_menu_id = menu_options[selected_menu_label]

        if st.button("Загрузить позиции меню", type="primary", key="c_btn_load_menu"):
            try:
                tok = st.session_state["transport_token"]
                org_id = st.session_state["transport_selected_org_id"]
                with st.spinner("Загрузка позиций внешнего меню..."):
                    menu_data = iiko_transport_menu_items(tok, selected_menu_id, org_id)
                ext_ids, ext_names = extract_external_menu_ids(menu_data)
                st.session_state["external_menu_ids"] = ext_ids
                st.session_state["external_menu_names"] = ext_names
                # Считаем уникальные блюда (itemId без sku дублей)
                n_items = sum(1 for k in ext_names if len(k) > 20)  # UUID длиннее 20
                n_skus = len(ext_names) - n_items
                st.success(f"Загружено: {n_skus} позиций (+ {n_items} UUID для матчинга)")
            except Exception as e:
                st.error(str(e))

    # Статус
    if "external_menu_ids" in st.session_state:
        ext_ids = st.session_state["external_menu_ids"]
        ext_names = st.session_state.get("external_menu_names", {})
        st.info(f"Фильтр по внешнему меню **активен**: {len(ext_ids)} ID. "
                "Рекомендации будут содержать только блюда из внешнего меню.")
        with st.expander("Позиции внешнего меню", expanded=False):
            menu_rows = []
            seen = set()
            for mid, mname in sorted(ext_names.items(), key=lambda x: x[1]):
                if mname not in seen:
                    menu_rows.append({"id": mid, "name": mname})
                    seen.add(mname)
            st.dataframe(pd.DataFrame(menu_rows), use_container_width=True)
        if st.button("Сбросить фильтр внешнего меню", key="c_btn_reset_ext"):
            for k in ["external_menu_ids", "external_menu_names"]:
                st.session_state.pop(k, None)
            st.rerun()


# ── Вкладка 2: Только iiko Server ──

with tab_api:
    st.subheader("Подключение к iiko Server")

    server_url2 = st.text_input(
        "URL сервера",
        placeholder="https://hostname:443",
        key="server_url",
    )

    col_l2, col_p2 = st.columns(2)
    with col_l2:
        login2 = st.text_input("Логин", key="login")
    with col_p2:
        password2 = st.text_input("Пароль", type="password", key="password")

    col_f2, col_t2 = st.columns(2)
    with col_f2:
        date_from2 = st.date_input(
            "Дата с",
            value=datetime.date.today() - datetime.timedelta(days=30),
            key="date_from",
        )
    with col_t2:
        date_to2 = st.date_input(
            "Дата по",
            value=datetime.date.today(),
            key="date_to",
        )

    if st.button("Загрузить данные с сервера", type="primary"):
        if not server_url2 or not login2 or not password2:
            st.error("Заполни URL сервера, логин и пароль.")
        else:
            try:
                with st.spinner("Аутентификация..."):
                    key = auth_iiko(server_url2, login2, password2)
                with st.spinner(f"Загрузка данных за {date_from2} – {date_to2}..."):
                    df_loaded = fetch_olap(server_url2, key, str(date_from2), str(date_to2))
                st.session_state["df_raw"] = df_loaded
                st.success(f"Получено строк: {len(df_loaded):,}")
            except requests.exceptions.HTTPError as e:
                st.error(f"Ошибка HTTP {e.response.status_code}: {e.response.text[:300]}")
            except requests.exceptions.ConnectionError:
                st.error("Не удалось подключиться к серверу. Проверь URL.")
            except Exception as e:
                st.error(str(e))


# ── Вкладка 3: XML / JSON файлы ──

with tab_xml:
    st.subheader("Загрузка файлов")

    st.markdown("##### Данные продаж (XML)")
    uploaded_xml = st.file_uploader("XML выгрузка iiko (REPORT_*.xml)", type=["xml"], key="upload_xml")
    if uploaded_xml:
        try:
            df_loaded = parse_iiko_report_xml(uploaded_xml.read())
            st.session_state["df_raw"] = df_loaded
            st.success(f"Распарсил строк: {len(df_loaded):,}")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.markdown("##### Внешнее меню (JSON из iikoTransport)")
    st.caption("Файл из API /api/2/menu/by_id")
    uploaded_menu = st.file_uploader("JSON внешнего меню", type=["json"], key="upload_menu_json")
    if uploaded_menu:
        try:
            menu_data = json.loads(uploaded_menu.read())
            ext_ids, ext_names = extract_external_menu_ids(menu_data)
            st.session_state["external_menu_ids"] = ext_ids
            st.session_state["external_menu_names"] = ext_names
            n_unique = len({v for v in ext_names.values()})
            st.success(f"Загружено позиций: {n_unique} (ID для матчинга: {len(ext_ids)})")
        except Exception as e:
            st.error(str(e))

    st.markdown("##### Номенклатура iiko Server (JSON, опционально)")
    st.caption("Для дополнительного матчинга UUID ↔ артикул")
    uploaded_nom = st.file_uploader("JSON номенклатуры", type=["json"], key="upload_nom_json")
    if uploaded_nom:
        try:
            nom_data = json.loads(uploaded_nom.read())
            id_to_num = parse_nomenclature_json(nom_data)
            st.session_state["nomenclature_id_to_num"] = id_to_num
            st.success(f"Номенклатура: {len(id_to_num)} позиций (UUID → артикул)")
        except Exception as e:
            st.error(str(e))


# ─── Общий пайплайн рекомендаций ─────────────────────────────────────────────

if "df_raw" not in st.session_state:
    st.info("Подключись к iiko Server или загрузи файлы — и я построю таблицу рекомендаций + дам CSV.")
    st.stop()

df_raw = st.session_state["df_raw"]

# ── Диагностика загруженных данных ──
with st.expander("🔍 Диагностика загруженных данных", expanded=False):
    st.write(f"**Строк:** {len(df_raw):,} | **Колонки:** `{list(df_raw.columns)}`")
    st.dataframe(df_raw.head(10), use_container_width=True)

    # Считаем сколько уникальных заказов и сколько из них с 2+ блюдами
    order_col = "UniqOrderId.Id" if "UniqOrderId.Id" in df_raw.columns else None
    dish_col = "DishId" if "DishId" in df_raw.columns else None
    if order_col and dish_col:
        orders_per_dish_count = df_raw.groupby(order_col)[dish_col].nunique()
        total_orders = len(orders_per_dish_count)
        multi_dish_orders = (orders_per_dish_count >= 2).sum()
        st.write(f"**Уникальных заказов:** {total_orders:,}")
        st.write(f"**Заказов с 2+ разными блюдами:** {multi_dish_orders:,}")
        if multi_dish_orders == 0:
            st.error("Нет ни одного заказа с 2+ блюдами — co-occurrence невозможен.")
    else:
        missing = [f for f in ["UniqOrderId.Id", "DishId"] if f not in df_raw.columns]
        st.error(f"Не найдены ожидаемые колонки: {missing}. iiko вернул другие имена.")

    if "DishGroup.SecondParent" in df_raw.columns:
        cats = df_raw["DishGroup.SecondParent"].value_counts().head(20)
        st.write("**Топ категорий в данных:**")
        st.dataframe(cats.rename("кол-во строк"), use_container_width=True)

# ── Подготовка allowed_ids для фильтрации ──
allowed_ids: set[str] | None = None
if "external_menu_ids" in st.session_state:
    allowed_ids = set(st.session_state["external_menu_ids"])

    # Если есть номенклатура — добавляем конвертацию UUID→артикул
    if "nomenclature_id_to_num" in st.session_state:
        id_to_num = st.session_state["nomenclature_id_to_num"]
        extra_ids = set()
        for mid in list(allowed_ids):
            if mid in id_to_num:
                extra_ids.add(id_to_num[mid])
        allowed_ids |= extra_ids

    # Диагностика матчинга
    if dish_col and dish_col in df_raw.columns:
        olap_dish_ids = set(df_raw[dish_col].astype(str).unique())
        matched = olap_dish_ids & allowed_ids
        st.info(
            f"Фильтр по внешнему меню: {len(allowed_ids)} ID в фильтре, "
            f"{len(olap_dish_ids)} уникальных DishId в OLAP, "
            f"**{len(matched)} совпадений**"
        )
        if len(matched) == 0:
            st.warning(
                "Нет совпадений между DishId из OLAP и ID внешнего меню! "
                "Проверь: DishId из OLAP может быть UUID, а SKU внешнего меню — артикул. "
                "Попробуй загрузить JSON номенклатуры для конвертации."
            )
            with st.expander("Примеры DishId из OLAP vs ID внешнего меню"):
                st.write("**DishId из OLAP (первые 10):**", list(olap_dish_ids)[:10])
                st.write("**ID из внешнего меню (первые 10):**", list(allowed_ids)[:10])

wide_df, long_df = build_recommendations(
    df_raw.copy(),
    top_n=int(top_n),
    min_co=int(min_co),
    excluded_categories=excluded_categories,
    allowed_ids=allowed_ids,
)

if long_df.empty:
    st.warning("По текущим настройкам рекомендаций не получилось (нет совместных продаж или всё отфильтровано).")
    st.info("Разверни блок «Диагностика» выше, чтобы понять причину.")
    st.stop()

# Filters UI
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    q = st.text_input("Поиск по названию блюда", value="")
with col2:
    categories = sorted(set(wide_df["category"].dropna().astype(str)))
    cat = st.selectbox("Категория", options=["(Все)"] + categories, index=0)
with col3:
    limit = st.number_input("Показать строк", min_value=10, value=200, step=10)

filtered = wide_df.copy()
if q.strip():
    filtered = filtered[filtered["dish_name"].str.contains(q.strip(), case=False, na=False)]
if cat != "(Все)":
    filtered = filtered[filtered["category"] == cat]

st.subheader("Таблица рекомендаций (блюдо → список)")
st.dataframe(filtered.head(int(limit)), use_container_width=True)

# CSV export
csv_buf = io.StringIO()
long_df.to_csv(csv_buf, index=False)
filename = f"recommendations_{datetime.date.today()}.csv"
st.download_button(
    label="⬇️ Скачать CSV для техподдержки",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name=filename,
    mime="text/csv",
    type="primary",
)

with st.expander("Показать CSV-формат (первые 200 строк)"):
    st.dataframe(long_df.head(200), use_container_width=True)
