import io
import itertools
import xml.etree.ElementTree as ET
from collections import defaultdict

import pandas as pd
import streamlit as st


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


def build_recommendations(
    df: pd.DataFrame,
    top_n: int = 8,
    min_co: int = 1,
    excluded_categories: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
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


st.set_page_config(page_title="iiko Recommendations", layout="wide")
st.title("iiko: рекомендации “берут вместе” из XML")

with st.sidebar:
    st.header("Настройки")
    top_n = st.slider("Макс. рекомендаций на блюдо", min_value=1, max_value=8, value=8, step=1)
    min_co = st.number_input("Мин. совместных покупок (co-occurrence)", min_value=1, value=1, step=1)

    st.caption("Исключения (обычно модификаторы):")
    excluded_default = ["OD Модификаторы", "GHD Модификаторы"]
    excluded_text = st.text_area("Категории для исключения (по одной в строке)", value="\n".join(excluded_default))
    excluded_categories = {x.strip() for x in excluded_text.splitlines() if x.strip()}

uploaded = st.file_uploader("Загрузи XML выгрузку iiko (REPORT_*.xml)", type=["xml"])

if not uploaded:
    st.info("Загрузи XML — и я построю таблицу рекомендаций + дам CSV.")
    st.stop()

xml_bytes = uploaded.read()

try:
    df_raw = parse_iiko_report_xml(xml_bytes)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"Распарсил строки: {len(df_raw):,}")

wide_df, long_df = build_recommendations(
    df_raw,
    top_n=int(top_n),
    min_co=int(min_co),
    excluded_categories=excluded_categories
)

if long_df.empty:
    st.warning("По текущим настройкам рекомендаций не получилось (нет совместных продаж или всё отфильтровано).")
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

# CSV export (long format is best for downstream)
csv_df = long_df.copy()
csv_buf = io.StringIO()
csv_df.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Скачать CSV (dish -> recommended dish)",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="recommendations.csv",
    mime="text/csv"
)

with st.expander("Показать CSV-формат (первые 200 строк)"):
    st.dataframe(csv_df.head(200), use_container_width=True)