// Client logging: console + fire-and-forget POST /log so client and server
// traces join up via the backend request_id.
const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')
let lastRequestId = null

export function setRequestId(id) {
  lastRequestId = id
}

export function getRequestId() {
  return lastRequestId
}

function send(level, message, extra = {}) {
  const body = JSON.stringify({
    level,
    message,
    request_id: lastRequestId,
    ua: navigator.userAgent,
    ...extra
  })
  try {
    if (navigator.sendBeacon) {
      // text/plain keeps it a CORS-simple request (no preflight) cross-origin;
      // the server parses the JSON body regardless of content-type.
      navigator.sendBeacon(`${API_BASE}/log`, new Blob([body], { type: 'text/plain;charset=UTF-8' }))
      return
    }
  } catch (_) { /* fall through */ }
  fetch(`${API_BASE}/log`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body, keepalive: true }).catch(() => {})
}

export const logger = {
  info: (message, extra) => console.log('[sticker]', message, extra || ''),
  event: (message, extra) => { console.log('[sticker]', message, extra || ''); send('info', message, extra) },
  error: (message, extra) => { console.error('[sticker]', message, extra || ''); send('error', message, extra) }
}

window.addEventListener('error', (e) => {
  logger.error('window.onerror', { msg: e.message, src: e.filename, line: e.lineno, col: e.colno })
})
window.addEventListener('unhandledrejection', (e) => {
  logger.error('unhandledrejection', { reason: String(e.reason) })
})
