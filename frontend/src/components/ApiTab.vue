<script setup>
import { reactive, ref } from 'vue'
import { recommendations, transportMenus, transportOrganizations } from '../api'

const props = defineProps({
  settings: { type: Object, required: true },
})
const emit = defineEmits(['result'])

function isoDate(d) {
  return d.toISOString().slice(0, 10)
}

const server = reactive({
  url: '',
  login: '',
  password: '',
  dateFrom: isoDate(new Date(Date.now() - 30 * 24 * 3600 * 1000)),
  dateTo: isoDate(new Date()),
})

// Фильтр по внешнему меню iikoTransport (опционально)
const useMenuFilter = ref(false)
const transport = reactive({
  apiKey: '',
  organizations: [],
  organizationId: '',
  menus: [],
  menuId: '',
})

const loadingOrgs = ref(false)
const loadingMenus = ref(false)
const loading = ref(false)
const error = ref('')
const notice = ref('')

async function loadOrganizations() {
  error.value = ''
  notice.value = ''
  if (!transport.apiKey.trim()) {
    error.value = 'Укажи API-ключ iikoTransport.'
    return
  }
  loadingOrgs.value = true
  try {
    const data = await transportOrganizations(transport.apiKey.trim())
    transport.organizations = data.organizations
    transport.organizationId = data.organizations[0]?.id || ''
    transport.menus = []
    transport.menuId = ''
    notice.value = `Токен получен. Организаций: ${data.organizations.length}`
  } catch (e) {
    error.value = e.message
  } finally {
    loadingOrgs.value = false
  }
}

async function loadMenus() {
  error.value = ''
  notice.value = ''
  loadingMenus.value = true
  try {
    const data = await transportMenus(transport.apiKey.trim(), transport.organizationId)
    transport.menus = data.external_menus
    transport.menuId = data.external_menus[0]?.id || ''
    notice.value = transport.menus.length
      ? `Найдено внешних меню: ${transport.menus.length}`
      : 'Внешних меню не найдено для этой организации.'
  } catch (e) {
    error.value = e.message
  } finally {
    loadingMenus.value = false
  }
}

async function build() {
  error.value = ''
  notice.value = ''
  if (!server.url || !server.login || !server.password) {
    error.value = 'Заполни URL сервера, логин и пароль.'
    return
  }
  if (useMenuFilter.value && !transport.menuId) {
    error.value = 'Выбери внешнее меню (или отключи фильтр).'
    return
  }
  loading.value = true
  try {
    const body = {
      iiko_server: {
        url: server.url,
        login: server.login,
        password: server.password,
        date_from: server.dateFrom,
        date_to: server.dateTo,
      },
      settings: {
        top_n: props.settings.topN,
        min_co: props.settings.minCo,
        excluded_categories: props.settings.excludedText
          .split('\n')
          .map((x) => x.trim())
          .filter(Boolean),
      },
    }
    if (useMenuFilter.value) {
      body.transport = {
        api_key: transport.apiKey.trim(),
        organization_id: transport.organizationId,
        external_menu_id: transport.menuId,
      }
    }
    const data = await recommendations(body)
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
    <h3>Данные продаж (iiko Server)</h3>

    <label>URL сервера</label>
    <input v-model="server.url" type="text" placeholder="https://hostname:443" />

    <div class="row">
      <div>
        <label>Логин</label>
        <input v-model="server.login" type="text" />
      </div>
      <div>
        <label>Пароль</label>
        <input v-model="server.password" type="password" />
      </div>
    </div>

    <div class="row">
      <div>
        <label>Дата с</label>
        <input v-model="server.dateFrom" type="date" />
      </div>
      <div>
        <label>Дата по</label>
        <input v-model="server.dateTo" type="date" />
      </div>
    </div>

    <hr class="divider" />

    <h3>
      <label style="display: inline-flex; align-items: center; gap: 8px; margin: 0; font-size: 15px; color: var(--text)">
        <input v-model="useMenuFilter" type="checkbox" />
        Фильтр по внешнему меню (iikoTransport)
      </label>
    </h3>
    <p class="muted">
      Если включён — рекомендации будут содержать только позиции из внешнего меню.
    </p>

    <template v-if="useMenuFilter">
      <label>API-ключ iikoTransport (apiKey)</label>
      <div class="row">
        <input
          v-model="transport.apiKey"
          type="password"
          placeholder="37ac8a1d7eb9446281c4934c0ba8f3f3"
        />
        <button :disabled="loadingOrgs" style="flex: 0 0 auto" @click="loadOrganizations">
          <span v-if="loadingOrgs" class="spinner"></span>
          Получить организации
        </button>
      </div>

      <template v-if="transport.organizations.length">
        <label>Организация</label>
        <div class="row">
          <select v-model="transport.organizationId">
            <option v-for="o in transport.organizations" :key="o.id" :value="o.id">
              {{ o.name }} ({{ o.id.slice(0, 8) }}…)
            </option>
          </select>
          <button :disabled="loadingMenus" style="flex: 0 0 auto" @click="loadMenus">
            <span v-if="loadingMenus" class="spinner"></span>
            Загрузить меню
          </button>
        </div>
      </template>

      <template v-if="transport.menus.length">
        <label>Внешнее меню</label>
        <select v-model="transport.menuId">
          <option v-for="m in transport.menus" :key="m.id" :value="m.id">
            {{ m.name }} (id: {{ m.id }})
          </option>
        </select>
      </template>
    </template>

    <div v-if="notice" class="alert info">{{ notice }}</div>
    <div v-if="error" class="alert error">{{ error }}</div>

    <hr class="divider" />

    <button class="primary" :disabled="loading" @click="build">
      <span v-if="loading" class="spinner"></span>
      {{ loading ? 'Загрузка и расчёт…' : 'Построить рекомендации' }}
    </button>
  </div>
</template>
