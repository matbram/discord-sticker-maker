import { logger, setRequestId } from './logger.js'

// Kick off processing. `source` is { file } or { url }. `onUpload(fraction)` reports
// upload progress (0..1). Returns { job_id, request_id }.
export function startProcess(source, params, onUpload) {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('params', JSON.stringify(params))
    if (source.file) form.append('file', source.file)
    else if (source.url) form.append('url', source.url)

    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/process')
    if (xhr.upload && onUpload) {
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) onUpload(e.loaded / e.total) }
      xhr.upload.onload = () => onUpload(1)
    }
    xhr.onload = () => {
      let data = {}
      try { data = JSON.parse(xhr.responseText) } catch (_) { /* ignore */ }
      if (data.request_id) setRequestId(data.request_id)
      if (xhr.status >= 200 && xhr.status < 300) {
        logger.event('process.started', { job_id: data.job_id })
        resolve(data)
      } else {
        const err = new Error(data.error || 'Failed to start processing')
        err.requestId = data.request_id
        reject(err)
      }
    }
    xhr.onerror = () => reject(new Error('Network error during upload'))
    xhr.send(form)
  })
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
    errors += 1
    if (errors >= 3) {
      es.close()
      logger.error('sse.lost_connection', { jobId })
      onEvent({ type: 'error', error: 'Lost connection to the server (it may have restarted). Please try again.' })
    }
  }
  return () => { finished = true; es.close() }
}

// type: '' (back-compat first output), a type ("sticker"|"emoji"|"gif"), or "all" (zip).
export function resultUrl(jobId, { type = '', download = false, ver = '' } = {}) {
  const path = type ? `/api/result/${jobId}/${type}` : `/api/result/${jobId}`
  const q = download ? '?download=1' : `?v=${ver || jobId}`
  return `${path}${q}`
}
