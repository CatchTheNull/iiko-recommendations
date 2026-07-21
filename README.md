# iiko Recommendations

Сервис рекомендаций «берут вместе» на основе данных продаж iiko.
Бэкенд — Django + DRF, фронтенд — Vue 3 + Vite (бывшее Streamlit-приложение).

## Структура

```
backend/    Django-проект (API + отдача собранного фронтенда)
  config/           настройки, urls, wsgi
  recommendations/  core.py (логика), iiko_api.py (клиенты iiko), views.py (API)
frontend/   Vue 3 SPA (Vite)
```

## Запуск в разработке

Бэкенд (порт 8000):

```bash
pip install -r requirements.txt
cd backend
python manage.py runserver
```

Фронтенд (порт 5173, проксирует /api на 8000):

```bash
cd frontend
npm install
npm run dev
```

Открой http://localhost:5173.

## Продакшен

```bash
cd frontend && npm run build   # соберёт frontend/dist
cd ../backend
DJANGO_DEBUG=0 DJANGO_SECRET_KEY=... gunicorn config.wsgi:application --bind 0.0.0.0:8000
```

Django сам отдаёт собранный SPA из `frontend/dist` — отдельный веб-сервер для
фронтенда не нужен. Либо через Docker (фронтенд собирается внутри):

```bash
docker build -t iiko-reco .
docker run -p 8000:8000 -e DJANGO_SECRET_KEY=change-me iiko-reco
```

## API

| Метод | Путь | Что делает |
|---|---|---|
| GET | `/api/health` | Проверка живости (без авторизации) |
| POST | `/api/auth/login` | Вход: `{"login": "...", "password": "..."}` → сессионная cookie |
| POST | `/api/auth/logout` | Выход |
| GET | `/api/auth/me` | Статус сессии |
| POST | `/api/transport/organizations` | Список организаций iikoTransport |
| POST | `/api/transport/menus` | Список внешних меню организации |
| POST | `/api/recommendations` | Продажи из iiko Server (+ опц. фильтр по внешнему меню) → рекомендации |
| POST | `/api/recommendations/from-files` | То же из загруженных XML/JSON файлов (multipart) |

К обоим `recommendations`-эндпоинтам можно добавить `?format=csv` — вернётся
CSV-файл для техподдержки вместо JSON.

Все эндпоинты, кроме `/api/health` и `/api/auth/*`, требуют входа. При первом
входе UI показывает форму логина; сессия хранится в подписанной cookie 2 недели.
Для curl: залогинься с `-c cookies.txt`, дальше передавай `-b cookies.txt`.

### Пример: рекомендации с сервера iiko

```bash
curl -X POST "http://localhost:8000/api/recommendations" \
  -H "Content-Type: application/json" \
  -d '{
    "iiko_server": {
      "url": "https://hostname:443",
      "login": "user",
      "password": "secret",
      "date_from": "2026-06-20",
      "date_to": "2026-07-20"
    },
    "transport": {
      "api_key": "…",
      "organization_id": "…",
      "external_menu_id": "…"
    },
    "settings": {"top_n": 8, "min_co": 1, "excluded_categories": []}
  }'
```

`transport` опционален; `organization_id` и `external_menu_id` можно узнать через
`/api/transport/organizations` и `/api/transport/menus`.

### Пример: рекомендации из файлов

```bash
curl -X POST "http://localhost:8000/api/recommendations/from-files?format=csv" \
  -F "sales_xml=@REPORT_2026.xml" \
  -F "menu_json=@external_menu.json" \
  -F "top_n=8" -F "min_co=1" \
  -o recommendations.csv
```

`menu_json` (внешнее меню из `/api/2/menu/by_id`) и `nomenclature_json`
(номенклатура iiko Server для матчинга UUID ↔ артикул) — опциональны.

### Ответ (JSON)

```json
{
  "diagnostics": {
    "rows": 12345,
    "total_orders": 4321,
    "multi_dish_orders": 3200,
    "external_menu_filter": {"filter_ids": 120, "olap_dish_ids": 340, "matched": 118}
  },
  "dishes": [
    {"dish_id": "…", "dish_name": "…", "category": "…", "recommendations": "Имя (ID) | …"}
  ],
  "recommendations": [
    {"dish_id": "…", "dish_name": "…", "category": "…",
     "recommended_dish_id": "…", "recommended_dish_name": "…",
     "rank": 1, "co_occurrence": 42}
  ]
}
```

Если `diagnostics.multi_dish_orders` = 0 — co-occurrence невозможен (нет заказов
с 2+ блюдами). Если `external_menu_filter.matched` = 0 — DishId из OLAP не
совпадают с ID внешнего меню; загрузи номенклатуру для конвертации UUID → артикул.

## Переменные окружения

- `APP_LOGIN`, `APP_PASSWORD` — учётные данные для входа в приложение
  (по умолчанию `admin`/`admin` — обязательно смени в проде)
- `DJANGO_SECRET_KEY` — обязательно задать в проде (им подписываются сессионные cookie)
- `DJANGO_DEBUG` — `0` в проде (по умолчанию `1`)
- `DJANGO_ALLOWED_HOSTS` — список хостов через запятую (по умолчанию `*`)
- `IIKO_APP_ID`, `IIKO_CLIENT_SECRET` — константы приложения для v2-авторизации
  iikoTransport (есть значения по умолчанию)
