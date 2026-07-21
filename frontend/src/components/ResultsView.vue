<script setup>
import { computed, ref } from 'vue'

const props = defineProps({
  result: { type: Object, required: true },
})

const search = ref('')
const category = ref('')
const limit = ref(200)

const diagnostics = computed(() => props.result.diagnostics || {})

const categories = computed(() => {
  const set = new Set(
    (props.result.dishes || []).map((d) => d.category).filter(Boolean)
  )
  return [...set].sort((a, b) => a.localeCompare(b, 'ru'))
})

const filteredDishes = computed(() => {
  let rows = props.result.dishes || []
  const q = search.value.trim().toLowerCase()
  if (q) rows = rows.filter((d) => (d.dish_name || '').toLowerCase().includes(q))
  if (category.value) rows = rows.filter((d) => d.category === category.value)
  return rows.slice(0, limit.value)
})

const menuFilter = computed(() => diagnostics.value.external_menu_filter)

function csvEscape(v) {
  const s = String(v ?? '')
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s
}

function downloadCsv() {
  const cols = [
    'dish_id',
    'dish_name',
    'category',
    'recommended_dish_id',
    'recommended_dish_name',
    'rank',
    'co_occurrence',
  ]
  const lines = [cols.join(',')]
  for (const r of props.result.recommendations || []) {
    lines.push(cols.map((c) => csvEscape(r[c])).join(','))
  }
  const blob = new Blob(['﻿' + lines.join('\n')], {
    type: 'text/csv;charset=utf-8',
  })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `recommendations_${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}
</script>

<template>
  <div class="panel" style="margin-top: 20px">
    <h2>Результаты</h2>

    <div v-if="!(result.recommendations || []).length" class="alert error">
      По текущим настройкам рекомендаций не получилось (нет совместных продаж или
      всё отфильтровано). Смотри диагностику ниже.
    </div>

    <details>
      <summary>🔍 Диагностика загруженных данных</summary>
      <p>
        <b>Строк:</b> {{ (diagnostics.rows ?? 0).toLocaleString('ru') }}
        <template v-if="diagnostics.total_orders != null">
          | <b>Уникальных заказов:</b>
          {{ diagnostics.total_orders.toLocaleString('ru') }} |
          <b>Заказов с 2+ блюдами:</b>
          {{ diagnostics.multi_dish_orders.toLocaleString('ru') }}
        </template>
      </p>
      <div v-if="diagnostics.multi_dish_orders === 0" class="alert error">
        Нет ни одного заказа с 2+ блюдами — co-occurrence невозможен.
      </div>
      <div v-if="menuFilter" class="alert info">
        Фильтр по внешнему меню: {{ menuFilter.filter_ids }} ID в фильтре,
        {{ menuFilter.olap_dish_ids }} уникальных DishId в продажах,
        <b>{{ menuFilter.matched }} совпадений</b>
      </div>
      <div v-if="menuFilter && menuFilter.matched === 0" class="alert error">
        Нет совпадений между DishId из OLAP и ID внешнего меню! DishId из OLAP
        может быть UUID, а SKU внешнего меню — артикул. Попробуй загрузить JSON
        номенклатуры для конвертации.
      </div>
    </details>

    <template v-if="(result.recommendations || []).length">
      <div class="row" style="margin: 12px 0">
        <div style="flex: 2">
          <label>Поиск по названию блюда</label>
          <input v-model="search" type="text" />
        </div>
        <div>
          <label>Категория</label>
          <select v-model="category">
            <option value="">(Все)</option>
            <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
          </select>
        </div>
        <div>
          <label>Показать строк</label>
          <input v-model.number="limit" type="number" min="10" step="10" />
        </div>
      </div>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Блюдо</th>
              <th>Категория</th>
              <th>Рекомендации</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in filteredDishes" :key="d.dish_id">
              <td>
                {{ d.dish_name }}
                <div class="muted">{{ d.dish_id }}</div>
              </td>
              <td>{{ d.category }}</td>
              <td>{{ d.recommendations }}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p class="muted">
        Показано {{ filteredDishes.length }} из {{ (result.dishes || []).length }} блюд.
        Всего пар рекомендаций: {{ result.recommendations.length }}
      </p>

      <button class="primary" @click="downloadCsv">
        ⬇️ Скачать CSV для техподдержки
      </button>
    </template>
  </div>
</template>
