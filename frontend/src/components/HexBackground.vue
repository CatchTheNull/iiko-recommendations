<script setup>
// Анимированный неоморфный фон: сетка выпуклых шестиугольников на canvas.
// Плитки плавно появляются/исчезают кластерами — фон «пересобирается».
import { onBeforeUnmount, onMounted, ref } from 'vue'

const canvasRef = ref(null)

const SIDE = 46 // длина стороны шестиугольника, px
const HEX_W = SIDE * 2
const HEX_H = Math.sqrt(3) * SIDE
const FADE_DURATION = 1600 // мс на появление/исчезновение плитки
const RESHUFFLE_EVERY = 1300 // мс между «пересборками» кластеров
const CLUSTER_RADIUS = HEX_H * 2.1

let ctx = null
let sprite = null
let cells = []
let rafId = 0
let reshuffleTimer = 0
let lastTs = 0
let reducedMotion = false

function buildSprite(dpr) {
  // Спрайт одной выпуклой плитки рисуем один раз — в цикле только drawImage.
  const pad = 30
  const w = (HEX_W + pad * 2) * dpr
  const h = (HEX_H + pad * 2) * dpr
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  const g = c.getContext('2d')
  g.scale(dpr, dpr)
  g.translate(HEX_W / 2 + pad, HEX_H / 2 + pad)

  const path = new Path2D()
  for (let i = 0; i < 6; i++) {
    const ang = (Math.PI / 3) * i
    const x = SIDE * 0.94 * Math.cos(ang)
    const y = SIDE * 0.94 * Math.sin(ang)
    if (i === 0) path.moveTo(x, y)
    else path.lineTo(x, y)
  }
  path.closePath()

  // Тёмная тень снизу-справа
  g.shadowColor = 'rgba(163, 170, 182, 0.55)'
  g.shadowOffsetX = 9
  g.shadowOffsetY = 9
  g.shadowBlur = 16
  g.fillStyle = '#eceef1'
  g.fill(path)

  // Светлая подсветка сверху-слева
  g.shadowColor = 'rgba(255, 255, 255, 0.95)'
  g.shadowOffsetX = -7
  g.shadowOffsetY = -7
  g.shadowBlur = 14
  g.fill(path)

  // Поверхность плитки — лёгкий градиент
  g.shadowColor = 'transparent'
  const grad = g.createLinearGradient(-SIDE, -SIDE, SIDE, SIDE)
  grad.addColorStop(0, '#f5f6f8')
  grad.addColorStop(1, '#e8eaee')
  g.fillStyle = grad
  g.fill(path)

  return { canvas: c, pad }
}

function buildGrid(width, height) {
  cells = []
  const stepX = SIDE * 1.5
  const stepY = HEX_H
  const cols = Math.ceil(width / stepX) + 2
  const rows = Math.ceil(height / stepY) + 2

  for (let col = -1; col < cols; col++) {
    for (let row = -1; row < rows; row++) {
      const x = col * stepX
      const y = row * stepY + (col % 2 ? stepY / 2 : 0)
      const on = Math.random() < 0.45
      cells.push({ x, y, s: on ? 1 : 0, target: on ? 1 : 0 })
    }
  }

  // Один проход сглаживания, чтобы стартовый узор был кластерным, как на референсе
  const snapshot = cells.map((c) => c.target)
  cells.forEach((cell, i) => {
    let near = 0
    let nearOn = 0
    cells.forEach((other, j) => {
      if (i === j) return
      const d = Math.hypot(cell.x - other.x, cell.y - other.y)
      if (d < HEX_H * 1.2) {
        near++
        if (snapshot[j]) nearOn++
      }
    })
    if (near > 0) {
      const on = nearOn / near > 0.5 ? 1 : snapshot[i]
      cell.target = on
      cell.s = on
    }
  })
}

function reshuffle() {
  // Пара случайных кластеров меняет состояние — фон «пересобирается».
  for (let k = 0; k < 2; k++) {
    const center = cells[Math.floor(Math.random() * cells.length)]
    if (!center) return
    const clusterOn = Math.random() < 0.5
    cells.forEach((cell) => {
      const d = Math.hypot(cell.x - center.x, cell.y - center.y)
      if (d < CLUSTER_RADIUS && Math.random() < 0.8) {
        cell.target = clusterOn ? 1 : 0
      }
    })
  }
}

function easeInOut(t) {
  return t * t * (3 - 2 * t)
}

function frame(ts) {
  const canvas = canvasRef.value
  if (!canvas || !ctx) return
  const dt = lastTs ? ts - lastTs : 16
  lastTs = ts

  if (!reducedMotion) {
    reshuffleTimer += dt
    if (reshuffleTimer >= RESHUFFLE_EVERY) {
      reshuffleTimer = 0
      reshuffle()
    }
  }

  const step = dt / FADE_DURATION
  const dpr = window.devicePixelRatio || 1
  const w = canvas.width / dpr
  const h = canvas.height / dpr

  ctx.clearRect(0, 0, w, h)

  const { canvas: sp, pad } = sprite
  const spriteW = HEX_W + pad * 2
  const spriteH = HEX_H + pad * 2

  for (const cell of cells) {
    if (cell.s < cell.target) cell.s = Math.min(cell.target, cell.s + step)
    else if (cell.s > cell.target) cell.s = Math.max(cell.target, cell.s - step)
    if (cell.s <= 0.01) continue

    const a = easeInOut(cell.s)
    // Лёгкое «вырастание» плитки вместе с прозрачностью
    const scale = 0.9 + 0.1 * a
    const dw = spriteW * scale
    const dh = spriteH * scale
    ctx.globalAlpha = a
    ctx.drawImage(sp, cell.x - dw / 2, cell.y - dh / 2, dw, dh)
  }
  ctx.globalAlpha = 1

  rafId = requestAnimationFrame(frame)
}

function resize() {
  const canvas = canvasRef.value
  if (!canvas) return
  const dpr = window.devicePixelRatio || 1
  canvas.width = window.innerWidth * dpr
  canvas.height = window.innerHeight * dpr
  ctx = canvas.getContext('2d')
  ctx.scale(dpr, dpr)
  sprite = buildSprite(dpr)
  buildGrid(window.innerWidth, window.innerHeight)
}

onMounted(() => {
  reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
  resize()
  window.addEventListener('resize', resize)
  rafId = requestAnimationFrame(frame)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(rafId)
  window.removeEventListener('resize', resize)
})
</script>

<template>
  <canvas ref="canvasRef" class="hexbg" aria-hidden="true"></canvas>
</template>

<style scoped>
.hexbg {
  position: fixed;
  inset: 0;
  width: 100vw;
  height: 100vh;
  z-index: -1;
  pointer-events: none;
}
</style>
