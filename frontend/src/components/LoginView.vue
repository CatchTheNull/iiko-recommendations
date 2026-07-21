<script setup>
import { ref } from 'vue'
import { authLogin } from '../api'

const emit = defineEmits(['success'])

const login = ref('')
const password = ref('')
const loading = ref(false)
const error = ref('')

async function submit() {
  error.value = ''
  if (!login.value.trim() || !password.value) {
    error.value = 'Укажи логин и пароль.'
    return
  }
  loading.value = true
  try {
    await authLogin(login.value.trim(), password.value)
    emit('success')
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-wrap">
    <form class="login-card" @submit.prevent="submit">
      <div class="logo-badge">
        <svg viewBox="60 20 340 210" class="logo-mark" aria-hidden="true">
          <!-- Синее облако с петлёй -->
          <g fill="none" stroke="#1657f2" stroke-width="27" stroke-linecap="round">
            <path d="M310 196 H128 A46 46 0 1 1 156 108 A66 66 0 0 1 278 82" />
            <path d="M278 82 A42 42 0 1 0 306 156 A42 42 0 0 0 278 82 Q296 60 330 56" />
          </g>
          <!-- Синяя точка -->
          <circle cx="365" cy="51" r="15" fill="#1657f2" />
        </svg>
      </div>

      <div class="logo-title">INNO-CLOUDS</div>
      <div class="logo-subtitle">
        <span class="line line-blue"></span>
        <span>RECOMENDATION</span>
        <span class="line line-orange"></span>
      </div>

      <input
        v-model="login"
        class="pill"
        type="text"
        placeholder="Логин"
        autocomplete="username"
        autofocus
      />
      <input
        v-model="password"
        class="pill"
        type="password"
        placeholder="Пароль"
        autocomplete="current-password"
      />

      <div v-if="error" class="alert error">{{ error }}</div>

      <button class="pill-btn" type="submit" :disabled="loading">
        <span v-if="loading" class="spinner"></span>
        <span class="grad-text">{{ loading ? 'Вход…' : 'Войти' }}</span>
      </button>
    </form>
  </div>
</template>

<style scoped>
.login-wrap {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.login-card {
  width: 100%;
  max-width: 420px;
  padding: 40px 44px 44px;
  border-radius: 32px;
  background: var(--bg);
  box-shadow:
    18px 18px 40px rgba(163, 170, 182, 0.55),
    -14px -14px 32px rgba(255, 255, 255, 0.9);
  display: flex;
  flex-direction: column;
}

/* Логотип в выпуклом круге */
.logo-badge {
  width: 118px;
  height: 118px;
  border-radius: 50%;
  margin: 0 auto 18px;
  background: var(--bg);
  box-shadow: var(--raised-sm);
  display: flex;
  align-items: center;
  justify-content: center;
}

.logo-mark {
  width: 80px;
  height: 50px;
  display: block;
}

.logo-title {
  text-align: center;
  font-size: 26px;
  font-weight: 800;
  letter-spacing: 0.06em;
  color: #23262e;
}

.logo-subtitle {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin: 6px 0 26px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.38em;
  text-indent: 0.38em;
  color: #4a4f5c;
}

.line {
  height: 2px;
  width: 34px;
  border-radius: 1px;
}

.line-blue {
  background: linear-gradient(90deg, #2456f0, #7a2bf0);
}

.line-orange {
  background: #f97316;
}

/* Пилюльные поля как на референсе */
input.pill {
  border-radius: 999px;
  padding: 15px 24px;
  font-size: 15px;
  margin-bottom: 20px;
  box-shadow:
    inset 6px 6px 12px var(--shadow-dark),
    inset -6px -6px 12px var(--shadow-light);
}

input.pill:focus {
  box-shadow:
    inset 6px 6px 12px var(--shadow-dark),
    inset -6px -6px 12px var(--shadow-light),
    0 0 0 2px rgba(122, 43, 240, 0.25);
}

/* Пилюльная кнопка: выпуклая, цветной только текст */
.pill-btn {
  border-radius: 999px;
  padding: 15px 24px;
  font-size: 17px;
  font-weight: 700;
  margin-top: 8px;
  background: var(--bg);
  box-shadow: var(--raised-sm);
}

.grad-text {
  background: linear-gradient(90deg, #2456f0, #7a2bf0);
  background-clip: text;
  -webkit-background-clip: text;
  color: transparent;
}

.pill-btn:hover {
  filter: brightness(1.03);
}

.pill-btn:active {
  box-shadow: var(--inset-sm);
}

.pill-btn .spinner {
  border-color: rgba(122, 43, 240, 0.3);
  border-top-color: #7a2bf0;
}
</style>
