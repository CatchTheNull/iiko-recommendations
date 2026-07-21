<script setup>
import { ref } from 'vue'
import { recommendationsFromFiles } from '../api'

const props = defineProps({
  settings: { type: Object, required: true },
})
const emit = defineEmits(['result'])

const salesXml = ref(null)
const menuJson = ref(null)
const nomJson = ref(null)
const loading = ref(false)
const error = ref('')

function pick(refVar) {
  return (event) => {
    refVar.value = event.target.files[0] || null
  }
}

const pickSales = pick(salesXml)
const pickMenu = pick(menuJson)
const pickNom = pick(nomJson)

async function build() {
  error.value = ''
  if (!salesXml.value) {
    error.value = 'Загрузи XML файл с данными продаж.'
    return
  }
  loading.value = true
  try {
    const fd = new FormData()
    fd.append('sales_xml', salesXml.value)
    if (menuJson.value) fd.append('menu_json', menuJson.value)
    if (nomJson.value) fd.append('nomenclature_json', nomJson.value)
    fd.append('top_n', String(props.settings.topN))
    fd.append('min_co', String(props.settings.minCo))
    fd.append('excluded_categories', props.settings.excludedText)
    const data = await recommendationsFromFiles(fd)
    emit('result', data)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div>
    <h3>Данные продаж (XML)</h3>
    <p class="muted">XML выгрузка iiko (REPORT_*.xml)</p>
    <input type="file" accept=".xml" @change="pickSales" />

    <hr class="divider" />

    <h3>Внешнее меню (JSON из iikoTransport) — опционально</h3>
    <p class="muted">Файл из API /api/2/menu/by_id. Ограничивает рекомендации позициями внешнего меню.</p>
    <input type="file" accept=".json" @change="pickMenu" />

    <h3>Номенклатура iiko Server (JSON) — опционально</h3>
    <p class="muted">Для дополнительного матчинга UUID ↔ артикул.</p>
    <input type="file" accept=".json" @change="pickNom" />

    <div v-if="error" class="alert error">{{ error }}</div>

    <hr class="divider" />

    <button class="primary" :disabled="loading" @click="build">
      <span v-if="loading" class="spinner"></span>
      {{ loading ? 'Расчёт…' : 'Построить рекомендации' }}
    </button>
  </div>
</template>
