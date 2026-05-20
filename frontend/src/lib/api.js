import { logger, setRequestId } from './logger.js'

// Kick off processing. `source` is { file } or { url }. Returns { job_id, request_id }.
export async function startProcess(source, params) {
  const form = new FormData()
  form.append('params', JSON.stringify(params))
  if (source.file) form.append('file', source.file)
  else if (source.url) form.append('url', source.url)

  const res = await fetch('/api/process', { method: 'POST', body: form })
  const data = await res.json().catch(() => ({}))
  if (data.request_id) setRequestId(data.request_id)
  if (!res.ok) {
    const err = new Error(data.error || 'Failed to start processing')
    err.requestId = data.request_id
    throw err
  }
  logger.event('process.started', { job_id: data.job_id })
  return data
}

// Subscribe to SSE progress. Calls onEvent(evt) for each event. Returns a close fn.
export function subscribeEvents(jobId, onEvent) {
  const es = new EventSource(`/api/events/${jobId}`)
  let finished = false
  let errors = 0
  es.onmessage = (e) => {
    let evt
    try { evt = JSON.parse(e.data) } catch (_) { return }
    errors = 0
    onEvent(evt)
    if (evt.type === 'result' || evt.type === 'error') { finished = true; es.close() }
  }
  es.onerror = () => {
    if (finished) { es.close(); return }
    // The browser auto-reconnects on transient drops; only give up after a few
    // failures (e.g. the server restarted/OOM'd and the job is gone).
    errors += 1
    if (errors >= 3) {
      es.close()
      logger.error('sse.lost_connection', { jobId })
      onEvent({ type: 'error', error: 'Lost connection to the server (it may have restarted). Please try again.' })
    }
  }
  return () => { finished = true; es.close() }
}

export function resultUrl(jobId, { download = false } = {}) {
  const q = download ? '?download=1' : `?t=${Date.now()}`
  return `/api/result/${jobId}${q}`
}
