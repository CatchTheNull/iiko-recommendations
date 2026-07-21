"""Чистая логика рекомендаций «берут вместе» — без зависимостей от UI/HTTP."""
from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd

# Поведение pandas 3.0; убирает FutureWarning о chained assignment в логах.
pd.set_option("mode.copy_on_write", True)

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
        children = list(el)
        if len(children) < 2:
            continue

        child_tags = {c.tag for c in children}
        if any(f in child_tags for f in REQUIRED_FIELDS):
            row = {}
            for c in children:
                if c.tag in REQUIRED_FIELDS:
                    row[c.tag] = _text(c)
            if row.get("UniqOrderId.Id") and row.get("DishId") and row.get("DishName"):
                rows.append(row)

    if rows:
        return pd.DataFrame(rows)

    # Strategy B: "field/value" style (e.g., <field name="DishId">..</field>)
    rows = []
    for el in root.iter():
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

    raise ValueError(
        "Не смог распарсить XML: не нашёл строки с полями UniqOrderId.Id, DishId, DishName. "
        "Возможен нестандартный формат выгрузки (например Spreadsheet XML)."
    )


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
            item_id = item.get("itemId", "")
            if item_id:
                all_ids.add(str(item_id))
                id_to_name[str(item_id)] = name
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


def expand_allowed_ids(allowed_ids: set[str], id_to_num: dict[str, str]) -> set[str]:
    """Дополняет множество ID внешнего меню артикулами из номенклатуры (UUID → num)."""
    extra = {id_to_num[mid] for mid in allowed_ids if mid in id_to_num}
    return allowed_ids | extra


def data_diagnostics(df: pd.DataFrame, allowed_ids: set[str] | None = None) -> dict:
    """Диагностика загруженных данных: заказы, матчинг с внешним меню."""
    diag: dict = {
        "rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
    }
    order_col = "UniqOrderId.Id" if "UniqOrderId.Id" in df.columns else None
    dish_col = "DishId" if "DishId" in df.columns else None
    if order_col and dish_col:
        orders_per_dish_count = df.groupby(order_col)[dish_col].nunique()
        diag["total_orders"] = int(len(orders_per_dish_count))
        diag["multi_dish_orders"] = int((orders_per_dish_count >= 2).sum())
    else:
        diag["missing_columns"] = [f for f in ["UniqOrderId.Id", "DishId"] if f not in df.columns]

    if allowed_ids is not None and dish_col:
        olap_dish_ids = set(df[dish_col].astype(str).unique())
        matched = olap_dish_ids & allowed_ids
        diag["external_menu_filter"] = {
            "filter_ids": len(allowed_ids),
            "olap_dish_ids": len(olap_dish_ids),
            "matched": len(matched),
        }
    return diag


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

    if long_df.empty:
        wide_df = pd.DataFrame(columns=["dish_id", "dish_name", "category", "recommendations"])
        return wide_df, long_df

    # Wide table: group recs into a single string list
    def _format_recs(g: pd.DataFrame) -> str:
        g2 = g.sort_values("rank")
        items = [f"{row.recommended_dish_name} ({row.recommended_dish_id})" for row in g2.itertuples()]
        return " | ".join(items)

    wide_df = (long_df
               .groupby(["dish_id", "dish_name", "category"], as_index=False)
               .apply(lambda g: pd.Series({"recommendations": _format_recs(g)}))
               .reset_index(drop=True))

    wide_df["rec_count"] = wide_df["recommendations"].apply(lambda s: 0 if not s else s.count("|") + 1)
    wide_df = wide_df.sort_values(["rec_count", "dish_name"], ascending=[False, True]).drop(columns=["rec_count"])

    return wide_df, long_df
