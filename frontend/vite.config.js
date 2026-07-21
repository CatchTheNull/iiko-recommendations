import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: {
      // В dev-режиме запросы к API проксируются на Django (localhost:8000)
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
