<script setup>
import { onMounted, reactive, ref } from 'vue'
import { authLogout, authMe } from './api'
import ApiTab from './components/ApiTab.vue'
import FilesTab from './components/FilesTab.vue'
import HexBackground from './components/HexBackground.vue'
import LoginView from './components/LoginView.vue'
import ResultsView from './components/ResultsView.vue'

const settings = reactive({
  topN: 8,
  minCo: 1,
  excludedText: '',
})

const tab = ref('api')
const result = ref(null)

// Авторизация при первом входе: пока не проверили сессию — ничего не показываем.
const authChecked = ref(false)
const authenticated = ref(false)

onMounted(async () => {
  try {
    const me = await authMe()
    authenticated.value = me.authenticated
  } catch {
    authenticated.value = false
  } finally {
    authChecked.value = true
  }
})

async function logout() {
  try {
    await authLogout()
  } finally {
    authenticated.value = false
    result.value = null
  }
}

function onResult(data) {
  result.value = data
}
</script>

<template>
  <HexBackground />

  <template v-if="!authChecked"></template>

  <LoginView v-else-if="!authenticated" @success="authenticated = true" />

  <div v-else class="layout">
    <aside class="panel sidebar">
      <h2>Настройки</h2>

      <label>Макс. рекомендаций на блюдо</label>
      <input v-model.number="settings.topN" type="number" min="1" max="50" />

      <label>Мин. совместных покупок (co-occurrence)</label>
      <input v-model.number="settings.minCo" type="number" min="1" />

      <label>Категории для исключения (по одной в строке)</label>
      <textarea
        v-model="settings.excludedText"
        placeholder="Обычно модификаторы"
      ></textarea>
    </aside>

    <main>
      <header class="page-header">
        <div>
          <h1>iiko: рекомендации «берут вместе»</h1>
          <p class="muted">
            Загрузи данные продаж — и получишь таблицу рекомендаций + CSV для техподдержки.
          </p>
        </div>
        <button class="logout-btn" title="Выйти из аккаунта" @click="logout">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
               stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
            <polyline points="16 17 21 12 16 7" />
            <line x1="21" y1="12" x2="9" y2="12" />
          </svg>
          Выйти
        </button>
      </header>

      <div class="panel">
        <div class="tabs">
          <button :class="{ active: tab === 'api' }" @click="tab = 'api'">
            🔌 iiko Server + iikoTransport
          </button>
          <button :class="{ active: tab === 'files' }" @click="tab = 'files'">
            📂 XML / JSON файлы
          </button>
        </div>

        <ApiTab v-show="tab === 'api'" :settings="settings" @result="onResult" />
        <FilesTab v-show="tab === 'files'" :settings="settings" @result="onResult" />
      </div>

      <ResultsView v-if="result" :result="result" />
    </main>
  </div>
</template>
