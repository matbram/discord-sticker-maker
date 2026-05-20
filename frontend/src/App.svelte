<script>
  import { onMount } from 'svelte'
  import { startProcess, subscribeEvents, resultUrl } from './lib/api.js'
  import { logger } from './lib/logger.js'

  const STEPS = [
    { key: 'decode', label: 'Decode' },
    { key: 'bg', label: 'Cutout' },
    { key: 'crop', label: 'Fit 320' },
    { key: 'encode', label: 'Optimize' }
  ]

  function defaultParams() {
    return {
      remove_bg: true,
      bg_model: 'auto',
      auto_crop: true,
      zoom: 1.0,
      offset_x: 0.0,
      offset_y: 0.0,
      padding: 0.06,
      max_fps: 24,
      max_duration_s: 4.0,
      trim_start_s: 0.0,
      max_bytes: 512000,
      max_colors: 256
    }
  }

  let view = 'idle' // idle | processing | done | error
  let params = defaultParams()
  let source = null // { file } | { url }
  let sourceName = ''
  let urlInput = ''
  let dragging = false
  let advancedOpen = false

  let jobId = null
  let progress = { stage: '', message: '', done: null, total: null }
  let meta = null
  let previewSrc = ''
  let previewBg = 'checker' // checker | dark | light
  let error = { message: '', requestId: '' }
  let closeStream = null

  let fileInput

  onMount(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('dsm_params') || 'null')
      if (saved) params = { ...defaultParams(), ...saved }
    } catch (_) { /* ignore */ }
  })

  function persistParams() {
    try { localStorage.setItem('dsm_params', JSON.stringify(params)) } catch (_) {}
  }

  function fmtBytes(n) {
    if (n == null) return '—'
    return n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`
  }

  function pickFiles(fileList) {
    const file = [...(fileList || [])].find((f) => f && f.size > 0)
    if (!file) return
    source = { file }
    sourceName = file.name || 'pasted image'
    start()
  }

  function submitUrl() {
    const url = urlInput.trim()
    if (!url) return
    source = { url }
    sourceName = url
    start()
  }

  async function start() {
    if (!source) return
    if (closeStream) closeStream()
    persistParams()
    view = 'processing'
    progress = { stage: 'decode', message: 'Starting…', done: null, total: null }
    meta = null
    error = { message: '', requestId: '' }
    logger.event('sticker.start', { name: sourceName, animatedHint: undefined })

    try {
      const { job_id } = await startProcess(source, params)
      jobId = job_id
      closeStream = subscribeEvents(job_id, onEvent)
    } catch (e) {
      view = 'error'
      error = { message: e.message || 'Something went wrong', requestId: e.requestId || '' }
      logger.error('sticker.start_failed', { error: String(e) })
    }
  }

  function onEvent(evt) {
    if (evt.type === 'progress') {
      progress = { stage: evt.stage, message: evt.message, done: evt.done, total: evt.total }
    } else if (evt.type === 'result') {
      meta = evt.meta
      previewSrc = resultUrl(jobId)
      view = 'done'
      logger.event('sticker.done', { bytes: meta.bytes, frames: meta.frames, format: meta.format })
    } else if (evt.type === 'error') {
      view = 'error'
      error = { message: evt.error || 'Processing failed', requestId: evt.request_id || '' }
      logger.error('sticker.failed', { error: evt.error })
    }
  }

  function regenerate() { start() }

  function reset() {
    if (closeStream) closeStream()
    view = 'idle'
    source = null
    sourceName = ''
    urlInput = ''
    meta = null
    previewSrc = ''
    jobId = null
  }

  function download() {
    if (!jobId) return
    const a = document.createElement('a')
    a.href = resultUrl(jobId, { download: true })
    a.download = 'my-sticker.png'
    document.body.appendChild(a)
    a.click()
    a.remove()
    logger.event('sticker.download')
  }

  // ---- global drag & paste ----
  function onDragOver(e) { e.preventDefault(); dragging = true }
  function onDragLeave(e) { if (e.relatedTarget === null) dragging = false }
  function onDrop(e) {
    e.preventDefault()
    dragging = false
    if (e.dataTransfer?.files?.length) pickFiles(e.dataTransfer.files)
    else {
      const text = e.dataTransfer?.getData('text')
      if (text && /^https?:\/\//i.test(text)) { urlInput = text; submitUrl() }
    }
  }
  function onPaste(e) {
    if (view === 'processing') return
    const items = e.clipboardData?.items || []
    for (const it of items) {
      if (it.type && it.type.startsWith('image/')) { pickFiles([it.getAsFile()]); return }
    }
    const text = e.clipboardData?.getData('text')
    if (text && /^https?:\/\//i.test(text.trim())) { urlInput = text.trim(); submitUrl() }
  }

  $: stepIndex = STEPS.findIndex((s) => s.key === progress.stage)
  $: pct = progress.total ? Math.round((progress.done / progress.total) * 100) : null
  $: checklist = meta?.checklist || {}
</script>

<svelte:window on:dragover={onDragOver} on:dragleave={onDragLeave} on:drop={onDrop} on:paste={onPaste} />

{#if dragging}
  <div class="drag-veil"><div class="drag-veil-inner">Drop to make a sticker</div></div>
{/if}

<header class="topbar">
  <div class="brand">
    <span class="brand-mark">◈</span>
    <span class="brand-name">Sticker Maker</span>
  </div>
  {#if view !== 'idle'}
    <button class="ghost-btn" on:click={reset}>＋ New sticker</button>
  {/if}
</header>

<main class="wrap">
  {#if view === 'idle'}
    <section class="hero">
      <h1>Turn <span class="grad">anything</span> into a Discord sticker</h1>
      <p class="sub">Drop an image, GIF, or video. We remove the background, crop it, and size it to Discord's spec — in seconds.</p>

      <button
        class="dropzone"
        class:drag={dragging}
        on:click={() => fileInput.click()}
        aria-label="Upload a file"
      >
        <div class="dz-icon">⬆</div>
        <div class="dz-title">Drop a file, click to browse, or paste</div>
        <div class="dz-sub">Images · GIFs · Video — any format</div>
      </button>
      <input
        bind:this={fileInput}
        type="file"
        accept="image/*,video/*"
        hidden
        on:change={(e) => pickFiles(e.target.files)}
      />

      <div class="or">or paste a link</div>
      <form class="url-row" on:submit|preventDefault={submitUrl}>
        <input type="url" placeholder="https://…/image.png" bind:value={urlInput} />
        <button type="submit" class="primary-btn" disabled={!urlInput.trim()}>Make sticker</button>
      </form>

      <div class="chips">
        <span class="chip">320×320 auto-fit</span>
        <span class="chip">Background removed</span>
        <span class="chip">Under 512&nbsp;KB</span>
        <span class="chip">PNG / APNG</span>
      </div>
    </section>

  {:else if view === 'processing'}
    <section class="card processing">
      <div class="spinner" aria-hidden="true"></div>
      <h2>Making your sticker…</h2>
      <p class="proc-msg">{progress.message || 'Working…'}</p>

      <div class="steps">
        {#each STEPS as step, i}
          <div class="step" class:active={i === stepIndex} class:done={stepIndex > i}>
            <span class="dot"></span>{step.label}
          </div>
        {/each}
      </div>

      <div class="bar" class:indeterminate={pct == null}>
        <div class="bar-fill" style={pct != null ? `width:${pct}%` : ''}></div>
      </div>
      {#if pct != null}<div class="bar-label">{progress.done} / {progress.total}</div>{/if}
    </section>

  {:else if view === 'done'}
    <section class="result">
      <div class="preview-panel">
        <div class="seg">
          <button class:on={previewBg === 'checker'} on:click={() => (previewBg = 'checker')}>Transparent</button>
          <button class:on={previewBg === 'dark'} on:click={() => (previewBg = 'dark')}>Dark</button>
          <button class:on={previewBg === 'light'} on:click={() => (previewBg = 'light')}>Light</button>
        </div>
        <div class="stage {previewBg}">
          <img class="sticker" src={previewSrc} alt="Your sticker preview" />
          <div class="render-hint">renders ~160px in chat</div>
        </div>
        <div class="badges">
          <span class="badge">{meta.width}×{meta.height}</span>
          <span class="badge" class:bad={!meta.under_limit}>{fmtBytes(meta.bytes)}</span>
          <span class="badge">{meta.format}</span>
          {#if meta.animated}<span class="badge">{meta.frames} frames{meta.fps ? ` · ${meta.fps}fps` : ''}</span>{/if}
        </div>
        <button class="primary-btn big" on:click={download}>⬇ Download sticker</button>
      </div>

      <div class="side-panel">
        <div class="checklist">
          <h3>Ready for Discord</h3>
          <ul>
            <li class:ok={checklist.dimensions_320}><span class="tick">{checklist.dimensions_320 ? '✓' : '✕'}</span> 320×320 pixels</li>
            <li class:ok={checklist.under_512kb}><span class="tick">{checklist.under_512kb ? '✓' : '✕'}</span> Under 512&nbsp;KB</li>
            <li class:ok={checklist.format_ok}><span class="tick">{checklist.format_ok ? '✓' : '✕'}</span> PNG / APNG format</li>
            <li class:ok={checklist.transparent} class:soft={!checklist.transparent}><span class="tick">{checklist.transparent ? '✓' : '○'}</span> Transparent background</li>
          </ul>
        </div>

        <details class="advanced" bind:open={advancedOpen}>
          <summary>Adjust &amp; regenerate</summary>
          <div class="controls">
            <label class="row"><span>Remove background</span><input type="checkbox" bind:checked={params.remove_bg} /></label>
            <label class="row"><span>Model</span>
              <select bind:value={params.bg_model}>
                <option value="auto">Auto</option>
                <option value="birefnet-general">General (best edges)</option>
                <option value="isnet-anime">Anime / art</option>
                <option value="birefnet-portrait">Portrait</option>
                <option value="u2net">Fast</option>
              </select>
            </label>
            <label class="row"><span>Zoom {params.zoom.toFixed(2)}×</span><input type="range" min="0.5" max="3" step="0.05" bind:value={params.zoom} /></label>
            <label class="row"><span>Offset X</span><input type="range" min="-1" max="1" step="0.05" bind:value={params.offset_x} /></label>
            <label class="row"><span>Offset Y</span><input type="range" min="-1" max="1" step="0.05" bind:value={params.offset_y} /></label>
            <label class="row"><span>Padding</span><input type="range" min="0" max="0.4" step="0.01" bind:value={params.padding} /></label>
            {#if meta.animated || true}
              <div class="grp">Animation</div>
              <label class="row"><span>Max FPS {params.max_fps}</span><input type="range" min="5" max="60" step="1" bind:value={params.max_fps} /></label>
              <label class="row"><span>Duration {params.max_duration_s}s</span><input type="range" min="0.5" max="10" step="0.5" bind:value={params.max_duration_s} /></label>
              <label class="row"><span>Trim start {params.trim_start_s}s</span><input type="range" min="0" max="10" step="0.5" bind:value={params.trim_start_s} /></label>
              <label class="row"><span>Max colors {params.max_colors}</span><input type="range" min="16" max="256" step="16" bind:value={params.max_colors} /></label>
            {/if}
          </div>
          <button class="primary-btn" on:click={regenerate}>↻ Regenerate</button>
        </details>

        <div class="howto">
          <h3>Uploading to Discord</h3>
          <ol>
            <li>Server Settings → Stickers → Upload.</li>
            <li>You need the <b>Manage Expressions</b> permission.</li>
            <li>Give it a name and a related <b>emoji</b>.</li>
          </ol>
        </div>
      </div>
    </section>

  {:else if view === 'error'}
    <section class="card error">
      <div class="err-icon">!</div>
      <h2>That didn't work</h2>
      <p>{error.message}</p>
      {#if error.requestId}<p class="ref">Reference: <code>{error.requestId}</code></p>{/if}
      <div class="err-actions">
        <button class="primary-btn" on:click={reset}>Try another file</button>
        {#if source}<button class="ghost-btn" on:click={regenerate}>Retry</button>{/if}
      </div>
    </section>
  {/if}
</main>

<footer class="footer">Self-hosted · no tracking · your files are processed then discarded</footer>

<style>
  .topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 24px; max-width: 1080px; margin: 0 auto;
  }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; }
  .brand-mark { color: var(--accent); font-size: 22px; }
  .brand-name { letter-spacing: 0.2px; }
  .ghost-btn {
    color: var(--muted); padding: 8px 14px; border-radius: var(--radius);
    border: 1px solid var(--border); background: var(--surface); font-weight: 600;
    transition: all 0.15s ease;
  }
  .ghost-btn:hover { color: var(--text); border-color: var(--accent); }

  .wrap { max-width: 1080px; margin: 0 auto; padding: 12px 24px 60px; }

  /* hero */
  .hero { text-align: center; padding: 36px 0 0; }
  .hero h1 { font-size: clamp(30px, 5vw, 50px); font-weight: 800; margin: 0 0 12px; letter-spacing: -1px; }
  .grad { background: linear-gradient(90deg, #5865f2, #8b5cf6); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .sub { color: var(--muted); max-width: 560px; margin: 0 auto 30px; font-size: 17px; }

  .dropzone {
    display: block; width: 100%; max-width: 620px; margin: 0 auto;
    background: var(--surface); border: 2px dashed var(--border);
    border-radius: var(--radius-lg); padding: 46px 24px; text-align: center;
    transition: all 0.18s ease;
  }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: var(--accent-soft); transform: translateY(-1px); }
  .dz-icon { font-size: 32px; color: var(--accent); margin-bottom: 8px; }
  .dz-title { font-weight: 600; font-size: 18px; }
  .dz-sub { color: var(--muted-2); margin-top: 6px; font-size: 14px; }

  .or { color: var(--muted-2); margin: 22px 0 12px; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
  .url-row { display: flex; gap: 10px; max-width: 620px; margin: 0 auto; }
  .url-row input {
    flex: 1; background: var(--bg-elevated); border: 1px solid var(--border);
    color: var(--text); padding: 12px 14px; border-radius: var(--radius); outline: none;
  }
  .url-row input:focus { border-color: var(--accent); box-shadow: var(--ring); }

  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 28px; }
  .chip { background: var(--surface); border: 1px solid var(--border); color: var(--muted); padding: 6px 12px; border-radius: 999px; font-size: 13px; }

  .primary-btn {
    background: var(--accent); color: #fff; font-weight: 600; padding: 12px 18px;
    border-radius: var(--radius); transition: background 0.15s ease, transform 0.05s ease;
  }
  .primary-btn:hover { background: var(--accent-hover); }
  .primary-btn:active { transform: translateY(1px); }
  .primary-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .primary-btn.big { width: 100%; padding: 14px; font-size: 16px; margin-top: 16px; }

  /* cards */
  .card {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg);
    padding: 40px; text-align: center; max-width: 560px; margin: 40px auto; box-shadow: var(--shadow);
  }
  .processing h2 { margin: 6px 0 4px; }
  .proc-msg { color: var(--muted); min-height: 22px; }
  .spinner {
    width: 46px; height: 46px; margin: 0 auto 12px; border-radius: 50%;
    border: 4px solid var(--border); border-top-color: var(--accent); animation: spin 0.8s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .steps { display: flex; justify-content: center; gap: 18px; margin: 22px 0 18px; flex-wrap: wrap; }
  .step { display: flex; align-items: center; gap: 7px; color: var(--muted-2); font-size: 14px; }
  .step .dot { width: 9px; height: 9px; border-radius: 50%; background: var(--border); }
  .step.active { color: var(--text); }
  .step.active .dot { background: var(--accent); box-shadow: 0 0 0 4px var(--accent-soft); }
  .step.done { color: var(--muted); }
  .step.done .dot { background: var(--success); }

  .bar { height: 8px; background: var(--bg-elevated); border-radius: 999px; overflow: hidden; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #5865f2, #8b5cf6); border-radius: 999px; transition: width 0.3s ease; }
  .bar.indeterminate .bar-fill { width: 35%; animation: slide 1.1s ease-in-out infinite; }
  @keyframes slide { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }
  .bar-label { color: var(--muted-2); font-size: 13px; margin-top: 8px; }

  /* result */
  .result { display: grid; grid-template-columns: 1.1fr 1fr; gap: 22px; margin-top: 24px; align-items: start; }
  @media (max-width: 820px) { .result { grid-template-columns: 1fr; } }

  .preview-panel, .side-panel > * {
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 18px;
  }
  .preview-panel { box-shadow: var(--shadow); }
  .side-panel { display: flex; flex-direction: column; gap: 16px; }

  .seg { display: inline-flex; background: var(--bg-elevated); border-radius: var(--radius); padding: 4px; gap: 4px; margin-bottom: 14px; }
  .seg button { padding: 7px 14px; border-radius: var(--radius-sm); color: var(--muted); font-weight: 600; font-size: 13px; }
  .seg button.on { background: var(--accent); color: #fff; }

  .stage {
    position: relative; border-radius: var(--radius); display: grid; place-items: center;
    padding: 26px; min-height: 280px;
  }
  .stage.dark { background: #313338; }
  .stage.light { background: #ffffff; }
  .stage.checker {
    background-color: #2b2d31;
    background-image:
      linear-gradient(45deg, #3a3c42 25%, transparent 25%),
      linear-gradient(-45deg, #3a3c42 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #3a3c42 75%),
      linear-gradient(-45deg, transparent 75%, #3a3c42 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0;
  }
  .sticker { width: 220px; height: 220px; image-rendering: auto; object-fit: contain; }
  .render-hint { position: absolute; bottom: 8px; right: 12px; font-size: 11px; color: var(--muted-2); background: rgba(0,0,0,0.35); padding: 2px 7px; border-radius: 6px; }
  .stage.light .render-hint { color: #6b7280; background: rgba(255,255,255,0.6); }

  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
  .badge { background: var(--bg-elevated); border: 1px solid var(--border); color: var(--muted); padding: 5px 11px; border-radius: 999px; font-size: 13px; font-weight: 600; }
  .badge.bad { color: var(--danger); border-color: var(--danger); }

  .checklist h3, .howto h3 { margin: 0 0 12px; font-size: 15px; }
  .checklist ul { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 9px; }
  .checklist li { color: var(--muted); display: flex; align-items: center; gap: 9px; font-size: 14px; }
  .checklist li.ok { color: var(--text); }
  .tick {
    display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 50%;
    background: var(--danger); color: #fff; font-size: 12px; font-weight: 700;
  }
  .checklist li.ok .tick { background: var(--success); }
  .checklist li.soft .tick { background: var(--muted-2); }

  .advanced summary { cursor: pointer; font-weight: 600; list-style: none; }
  .advanced summary::-webkit-details-marker { display: none; }
  .advanced summary::before { content: '▸ '; color: var(--accent); }
  .advanced[open] summary::before { content: '▾ '; }
  .controls { display: flex; flex-direction: column; gap: 11px; margin: 14px 0; }
  .grp { color: var(--muted-2); font-size: 12px; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px; }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 14px; color: var(--muted); }
  .row span { white-space: nowrap; }
  .row input[type='range'] { flex: 1; max-width: 56%; accent-color: var(--accent); }
  .row select { background: var(--bg-elevated); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; }
  .row input[type='checkbox'] { width: 18px; height: 18px; accent-color: var(--accent); }

  .howto ol { margin: 0; padding-left: 18px; color: var(--muted); font-size: 14px; display: flex; flex-direction: column; gap: 6px; }

  .error .err-icon { width: 48px; height: 48px; border-radius: 50%; background: var(--danger); color: #fff; display: grid; place-items: center; font-size: 26px; font-weight: 800; margin: 0 auto 10px; }
  .error .ref { color: var(--muted-2); font-size: 13px; }
  .error code { background: var(--bg-elevated); padding: 2px 7px; border-radius: 5px; }
  .err-actions { display: flex; gap: 10px; justify-content: center; margin-top: 16px; }

  .footer { text-align: center; color: var(--muted-2); font-size: 13px; padding: 24px; }

  .drag-veil { position: fixed; inset: 0; background: rgba(30,31,34,0.82); z-index: 50; display: grid; place-items: center; backdrop-filter: blur(2px); }
  .drag-veil-inner { border: 2px dashed var(--accent); border-radius: var(--radius-lg); padding: 50px 70px; font-size: 22px; font-weight: 700; color: #fff; }
</style>
