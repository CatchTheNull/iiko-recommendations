// Обёртки над API Django-бэкенда.

async function handleResponse(resp) {
  if (!resp.ok) {
    // Сессия истекла — перезагружаем страницу, появится экран входа.
    if ((resp.status === 401 || resp.status === 403) && !resp.url.includes('/api/auth/')) {
      window.location.reload()
      return new Promise(() => {})
    }
    let detail
    try {
      const data = await resp.json()
      detail = data.detail || JSON.stringify(data)
    } catch {
      detail = await resp.text()
    }
    throw new Error(detail || `HTTP ${resp.status}`)
  }
  return resp.json()
}

export function postJson(url, body) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handleResponse)
}

export function postForm(url, formData) {
  return fetch(url, { method: 'POST', body: formData }).then(handleResponse)
}

export function authMe() {
  return fetch('/api/auth/me').then(handleResponse)
}

export function authLogin(login, password) {
  return postJson('/api/auth/login', { login, password })
}

export function authLogout() {
  return fetch('/api/auth/logout', { method: 'POST' }).then(handleResponse)
}

export function transportOrganizations(apiKey) {
  return postJson('/api/transport/organizations', { api_key: apiKey })
}

export function transportMenus(apiKey, organizationId) {
  return postJson('/api/transport/menus', {
    api_key: apiKey,
    organization_id: organizationId,
  })
}

export function recommendations(body) {
  return postJson('/api/recommendations', body)
}

export function recommendationsFromFiles(formData) {
  return postForm('/api/recommendations/from-files', formData)
}
