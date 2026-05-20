<script>
  import { onMount } from 'svelte'
  import { startProcess, subscribeEvents, resultUrl } from './lib/api.js'
  import { logger } from './lib/logger.js'
  import Cropper from './lib/Cropper.svelte'
  import Trim from './lib/Trim.svelte'

  const STEPS = [
    { key: 'decode', label: 'Decode' },
    { key: 'bg', label: 'Cutout' },
    { key: 'crop', label: 'Fit 320' },
    { key: 'encode', label: 'Optimize' }
  ]

  // One-tap presets bundle settings so the user never thinks in parameters.
  const PRESETS = [
    { id: 'auto', emoji: '✨', label: 'Auto', patch: { remove_bg: false, priority: 'balanced', fit_mode: 'fit', max_colors: 256 } },
    { id: 'smoothest', emoji: '🎞️', label: 'Smoothest', patch: { remove_bg: false, priority: 'smooth', fit_mode: 'fit', max_colors: 128, max_fps: 30 } },
    { id: 'cutout', emoji: '🪄', label: 'Cut-out', patch: { remove_bg: true, bg_model: 'auto', priority: 'balanced', fit_mode: 'fit' } },
    { id: 'crisp', emoji: '💎', label: 'Crisp', patch: { remove_bg: false, priority: 'sharp', fit_mode: 'fit', max_colors: 256 } }
  ]

  function defaultParams() {
    return {
      remove_bg: false, bg_model: 'auto', auto_crop: true, fit_mode: 'fit',
      zoom: 1.0, offset_x: 0.0, offset_y: 0.0, padding: 0.06,
      max_fps: 18, max_duration_s: 4.0, trim_start_s: 0.0,
      priority: 'balanced', max_bytes: 512000, max_colors: 256
    }
  }

  let view = 'idle' // idle | processing | done | error
  let params = defaultParams()
  let source = null // { file } | { url }
  let sourceName = ''
  let urlInput = ''
  let dragging = false
  let advancedOpen = false
  let activePreset = 'auto'

  // client-side source preview (for the cropper)
  let sourceUrl = ''
  let sourceIsVideo = false
  let sourceW = 0
  let sourceH = 0

  let jobId = null
  let progress = { stage: '', message: '', done: null, total: null }
  let meta = null
  let previewSrc = ''
  let previewBg = 'checker'
  let error = { message: '', requestId: '' }
  let closeStream = null
  let watchdogTimer = null
  let busy = false
  let pendingRegen = false
  let regenTimer = null
  let fileInput

  const PARAMS_KEY = 'dsm_params_v2'
  onMount(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(PARAMS_KEY) || 'null')
      if (saved) params = { ...defaultParams(), ...saved }
    } catch (_) { /* ignore */ }
  })
  function persistParams() { try { localStorage.setItem(PARAMS_KEY, JSON.stringify(params)) } catch (_) {} }

  function armWatchdog() {
    clearTimeout(watchdogTimer)
    watchdogTimer = setTimeout(() => {
      if (busy) {
        if (closeStream) closeStream()
        busy = false
        view = 'error'
        error = { message: 'This is taking longer than expected — the server may have restarted. Please try again.', requestId: '' }
        logger.error('sticker.watchdog_timeout')
      }
    }, 90000)
  }
  function disarmWatchdog() { clearTimeout(watchdogTimer); watchdogTimer = null }

  function fmtBytes(n) { return n == null ? '—' : (n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`) }

  function clearSourceUrl() { if (sourceUrl && sourceUrl.startsWith('blob:')) try { URL.revokeObjectURL(sourceUrl) } catch (_) {} }
  function loadDims(url, isVideo) {
    sourceW = 0; sourceH = 0
    if (isVideo) {
      const v = document.createElement('video')
      v.onloadedmetadata = () => { sourceW = v.videoWidth; sourceH = v.videoHeight }
      v.src = url
    } else {
      const im = new Image()
      im.onload = () => { sourceW = im.naturalWidth; sourceH = im.naturalHeight }
      im.src = url
    }
  }

  function newSourceCommon() {
    // fresh framing for a new file; keep look prefs (priority/bg)
    params = { ...params, zoom: 1, offset_x: 0, offset_y: 0, trim_start_s: 0 }
    loadDims(sourceUrl, sourceIsVideo)
    firstRun()
  }
  function pickFiles(fileList) {
    const file = [...(fileList || [])].find((f) => f && f.size > 0)
    if (!file) return
    clearSourceUrl()
    source = { file }
    sourceName = file.name || 'pasted'
    sourceIsVideo = (file.type || '').startsWith('video/')
    sourceUrl = URL.createObjectURL(file)
    newSourceCommon()
  }
  function submitUrl() {
    const url = urlInput.trim()
    if (!url) return
    clearSourceUrl()
    source = { url }
    sourceName = url
    sourceIsVideo = /\.(mp4|mov|webm|mkv|avi)(\?|$)/i.test(url)
    sourceUrl = url
    newSourceCommon()
  }

  function firstRun() { view = 'processing'; run() }

  async function run() {
    if (!source) return
    if (closeStream) closeStream()
    persistParams()
    busy = true
    if (view !== 'done') { progress = { stage: 'decode', message: 'Starting…', done: null, total: null } }
    error = { message: '', requestId: '' }
    try {
      const { job_id } = await startProcess(source, params)
      jobId = job_id
      closeStream = subscribeEvents(job_id, onEvent)
      armWatchdog()
    } catch (e) {
      busy = false; disarmWatchdog()
      view = 'error'
      error = { message: e.message || 'Something went wrong', requestId: e.requestId || '' }
      logger.error('sticker.start_failed', { error: String(e) })
    }
  }

  // Debounced auto-update: the user just tweaks; we regenerate behind the scenes.
  function scheduleRegen() {
    persistParams()
    clearTimeout(regenTimer)
    regenTimer = setTimeout(() => { if (busy) pendingRegen = true; else run() }, 500)
  }

  function onEvent(evt) {
    armWatchdog()
    if (evt.type === 'progress') {
      progress = { stage: evt.stage, message: evt.message, done: evt.done, total: evt.total }
    } else if (evt.type === 'result') {
      disarmWatchdog(); busy = false
      meta = evt.meta
      previewSrc = resultUrl(jobId)
      view = 'done'
      logger.event('sticker.done', { bytes: meta.bytes, frames: meta.frames, fps: meta.fps })
      if (pendingRegen) { pendingRegen = false; run() }
    } else if (evt.type === 'error') {
      disarmWatchdog(); busy = false
      view = 'error'
      error = { message: evt.error || 'Processing failed', requestId: evt.request_id || '' }
      logger.error('sticker.failed', { error: evt.error })
    }
  }

  function applyPreset(p) { activePreset = p.id; params = { ...params, ...p.patch }; scheduleRegen() }
  function onCropChange(e) { params = { ...params, zoom: e.detail.zoom, offset_x: e.detail.offsetX, offset_y: e.detail.offsetY }; activePreset = ''; scheduleRegen() }
  function setFit(m) { params.fit_mode = m; activePreset = ''; scheduleRegen() }
  function setPriority(p) { params.priority = p; activePreset = ''; scheduleRegen() }
  function toggleBg() { params.remove_bg = !params.remove_bg; activePreset = ''; scheduleRegen() }
  function onTrimChange(e) { params = { ...params, trim_start_s: e.detail.start, max_duration_s: e.detail.length }; scheduleRegen() }
  function shortenClip() { params.max_duration_s = Math.max(1, Math.round((params.max_duration_s / 2) * 2) / 2); scheduleRegen() }

  function reset() {
    if (closeStream) closeStream()
    disarmWatchdog(); clearSourceUrl()
    busy = false; view = 'idle'; source = null; sourceName = ''; urlInput = ''
    sourceUrl = ''; sourceW = 0; sourceH = 0; meta = null; previewSrc = ''; jobId = null
  }

  function download() {
    if (!jobId) return
    const a = document.createElement('a')
    a.href = resultUrl(jobId, { download: true })
    a.download = 'my-sticker.png'
    document.body.appendChild(a); a.click(); a.remove()
    logger.event('sticker.download')
  }

  // ---- global drag & paste ----
  function onDragOver(e) { e.preventDefault(); dragging = true }
  function onDragLeave(e) { if (e.relatedTarget === null) dragging = false }
  function onDrop(e) {
    e.preventDefault(); dragging = false
    if (e.dataTransfer?.files?.length) pickFiles(e.dataTransfer.files)
    else { const t = e.dataTransfer?.getData('text'); if (t && /^https?:\/\//i.test(t)) { urlInput = t; submitUrl() } }
  }
  function onPaste(e) {
    if (busy) return
    const items = e.clipboardData?.items || []
    for (const it of items) { if (it.type && it.type.startsWith('image/')) { pickFiles([it.getAsFile()]); return } }
    const t = e.clipboardData?.getData('text')
    if (t && /^https?:\/\//i.test(t.trim())) { urlInput = t.trim(); submitUrl() }
  }

  $: stepIndex = STEPS.findIndex((s) => s.key === progress.stage)
  $: pct = progress.total ? Math.round((progress.done / progress.total) * 100) : null
  $: checklist = meta?.checklist || {}
  $: estFrames = Math.min(72, Math.max(1, Math.round(params.max_fps * params.max_duration_s)))
  $: animated = !!meta?.animated
  $: smoothWord = !animated ? '' : (meta.fps >= 15 ? 'Very smooth' : meta.fps >= 8 ? 'Smooth' : 'A little choppy')
  $: choppy = animated && (meta.fps || 0) < 8
</script>

<svelte:window on:dragover={onDragOver} on:dragleave={onDragLeave} on:drop={onDrop} on:paste={onPaste} />

{#if dragging}
  <div class="drag-veil"><div class="drag-veil-inner">Drop to make a sticker</div></div>
{/if}

<header class="topbar">
  <div class="brand"><span class="brand-mark">◈</span><span class="brand-name">Sticker Maker</span></div>
  {#if view !== 'idle'}<button class="ghost-btn" on:click={reset}>＋ New sticker</button>{/if}
</header>

<main class="wrap">
  {#if view === 'idle'}
    <section class="hero">
      <h1>Turn <span class="grad">anything</span> into a Discord sticker</h1>
      <p class="sub">Drop an image, GIF, or video. We size it to Discord's spec — then tweak it with one tap.</p>
      <button class="dropzone" class:drag={dragging} on:click={() => fileInput.click()} aria-label="Upload a file">
        <div class="dz-icon">⬆</div>
        <div class="dz-title">Drop a file, click to browse, or paste</div>
        <div class="dz-sub">Images · GIFs · Video — any format</div>
      </button>
      <input bind:this={fileInput} type="file" accept="image/*,video/*" hidden on:change={(e) => pickFiles(e.target.files)} />
      <div class="or">or paste a link</div>
      <form class="url-row" on:submit|preventDefault={submitUrl}>
        <input type="url" placeholder="https://…/image.png" bind:value={urlInput} />
        <button type="submit" class="primary-btn" disabled={!urlInput.trim()}>Make sticker</button>
      </form>
      <div class="chips">
        <span class="chip">320×320 auto-fit</span><span class="chip">Under 512&nbsp;KB</span>
        <span class="chip">One-tap presets</span><span class="chip">PNG / APNG</span>
      </div>
    </section>

  {:else if view === 'processing'}
    <section class="card processing">
      <div class="spinner" aria-hidden="true"></div>
      <h2>Making your sticker…</h2>
      <p class="proc-msg">{progress.message || 'Working…'}</p>
      <div class="steps">
        {#each STEPS as step, i}
          <div class="step" class:active={i === stepIndex} class:done={stepIndex > i}><span class="dot"></span>{step.label}</div>
        {/each}
      </div>
      <div class="bar" class:indeterminate={pct == null}><div class="bar-fill" style={pct != null ? `width:${pct}%` : ''}></div></div>
      {#if pct != null}<div class="bar-label">{progress.done} / {progress.total}</div>{/if}
    </section>

  {:else if view === 'done'}
    <section class="result">
      <!-- LEFT: the sticker you get -->
      <div class="preview-panel">
        <div class="seg">
          <button class:on={previewBg === 'checker'} on:click={() => (previewBg = 'checker')}>Transparent</button>
          <button class:on={previewBg === 'dark'} on:click={() => (previewBg = 'dark')}>Dark</button>
          <button class:on={previewBg === 'light'} on:click={() => (previewBg = 'light')}>Light</button>
        </div>
        <div class="stage {previewBg}">
          <img class="sticker" src={previewSrc} alt="Your sticker preview" />
          <div class="render-hint">renders ~160px in chat</div>
          {#if busy}<div class="busy"><div class="spinner sm"></div></div>{/if}
        </div>

        <div class="status">
          <span class="ready">✓ Ready for Discord</span>
          <span class="summary">{meta.format} · {fmtBytes(meta.bytes)}{animated ? ` · ${smoothWord} (${meta.fps} fps, ${meta.frames} frames)` : ''}</span>
        </div>

        {#if choppy}
          <div class="fixes">
            <button class="chip-btn" on:click={() => applyPreset(PRESETS[1])}>Add more frames</button>
            <button class="chip-btn" on:click={shortenClip}>Shorten the clip</button>
          </div>
        {/if}

        <button class="primary-btn big" on:click={download} disabled={busy}>⬇ Download sticker</button>
        <details class="mini"><summary>Details</summary>
          <div class="mini-grid">
            <span>{meta.width}×{meta.height}px</span><span>{fmtBytes(meta.bytes)}</span>
            <span>{meta.format}</span>{#if animated}<span>{meta.frames} frames · {meta.fps} fps</span>{/if}
          </div>
          {#if meta.notes && meta.notes.length}<ul class="notes">{#each meta.notes as n}<li>{n}</li>{/each}</ul>{/if}
        </details>
      </div>

      <!-- RIGHT: friendly editor -->
      <div class="side-panel">
        <div class="presets">
          {#each PRESETS as p}
            <button class="preset" class:on={activePreset === p.id} on:click={() => applyPreset(p)}>
              <span class="p-emoji">{p.emoji}</span><span>{p.label}</span>
            </button>
          {/each}
        </div>

        <div class="card-sm">
          <h3>Frame it</h3>
          {#if sourceUrl && sourceW > 0}
            <div class="crop-wrap">
              <Cropper src={sourceUrl} isVideo={sourceIsVideo} naturalW={sourceW} naturalH={sourceH}
                       zoom={params.zoom} offsetX={params.offset_x} offsetY={params.offset_y}
                       fitMode={params.fit_mode} padding={params.padding} on:change={onCropChange} />
            </div>
          {:else}
            <p class="muted-line">Loading preview…</p>
          {/if}
          <div class="seg small">
            <button class:on={params.fit_mode === 'fit'} on:click={() => setFit('fit')}>Fit</button>
            <button class:on={params.fit_mode === 'fill'} on:click={() => setFit('fill')}>Fill</button>
          </div>
          <label class="row"><span>Zoom</span>
            <input type="range" min="0.5" max="4" step="0.05" bind:value={params.zoom} on:change={() => { activePreset = ''; scheduleRegen() }} /></label>
        </div>

        {#if animated}
          <div class="card-sm">
            <h3>Frames vs colors{meta.frames ? ` — ${meta.frames} frames now` : ''}</h3>
            <div class="seg three">
              <button class:on={params.priority === 'smooth'} on:click={() => setPriority('smooth')}>More frames</button>
              <button class:on={params.priority === 'balanced'} on:click={() => setPriority('balanced')}>Balanced</button>
              <button class:on={params.priority === 'sharp'} on:click={() => setPriority('sharp')}>Richer colors</button>
            </div>
            <p class="muted-line">A 320×320 sticker must fit in 512 KB. <b>More frames</b> spends the budget on extra frames (fewer colors); <b>Richer colors</b> keeps colors but fewer frames.</p>
          </div>
        {/if}

        <div class="card-sm">
          <label class="toggle"><span>Cut out background</span>
            <input type="checkbox" checked={params.remove_bg} on:change={toggleBg} /></label>
        </div>

        {#if sourceIsVideo}
          <div class="card-sm">
            <h3>Pick the moment</h3>
            <Trim src={sourceUrl} start={params.trim_start_s} length={params.max_duration_s} on:change={onTrimChange} />
          </div>
        {/if}

        <details class="advanced card-sm" bind:open={advancedOpen}>
          <summary>Advanced</summary>
          <div class="controls">
            <label class="row"><span>Bg model</span>
              <select bind:value={params.bg_model} on:change={scheduleRegen}>
                <option value="auto">Auto</option><option value="birefnet-general">General</option>
                <option value="isnet-anime">Anime / art</option><option value="birefnet-portrait">Portrait</option>
                <option value="u2net">Fast</option>
              </select></label>
            <label class="row"><span>Padding</span><input type="range" min="0" max="0.4" step="0.01" bind:value={params.padding} on:change={scheduleRegen} /></label>
            {#if animated}
              <label class="row"><span>Max colors {params.max_colors}</span><input type="range" min="8" max="256" step="8" bind:value={params.max_colors} on:change={scheduleRegen} /></label>
              <div class="est">Fewer colors = more frames fit under 512 KB.</div>
            {/if}
            {#if sourceIsVideo}
              <label class="row"><span>Sample FPS {params.max_fps}</span><input type="range" min="5" max="60" step="1" bind:value={params.max_fps} on:change={scheduleRegen} /></label>
              <label class="row"><span>Duration {params.max_duration_s}s</span><input type="range" min="0.5" max="10" step="0.5" bind:value={params.max_duration_s} on:change={scheduleRegen} /></label>
              <div class="est">≈ {estFrames} frames sampled (then trimmed to fit)</div>
            {/if}
          </div>
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
        {#if source}<button class="ghost-btn" on:click={() => { view = 'processing'; run() }}>Retry</button>{/if}
      </div>
    </section>
  {/if}
</main>

<footer class="footer">Self-hosted · no tracking · your files are processed then discarded</footer>

<style>
  .topbar { display: flex; align-items: center; justify-content: space-between; padding: 18px 24px; max-width: 1080px; margin: 0 auto; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; }
  .brand-mark { color: var(--accent); font-size: 22px; }
  .ghost-btn { color: var(--muted); padding: 8px 14px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface); font-weight: 600; transition: all 0.15s ease; }
  .ghost-btn:hover { color: var(--text); border-color: var(--accent); }
  .wrap { max-width: 1080px; margin: 0 auto; padding: 12px 24px 60px; }

  .hero { text-align: center; padding: 36px 0 0; }
  .hero h1 { font-size: clamp(30px, 5vw, 50px); font-weight: 800; margin: 0 0 12px; letter-spacing: -1px; }
  .grad { background: linear-gradient(90deg, #5865f2, #8b5cf6); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .sub { color: var(--muted); max-width: 560px; margin: 0 auto 30px; font-size: 17px; }
  .dropzone { display: block; width: 100%; max-width: 620px; margin: 0 auto; background: var(--surface); border: 2px dashed var(--border); border-radius: var(--radius-lg); padding: 46px 24px; text-align: center; transition: all 0.18s ease; }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: var(--accent-soft); transform: translateY(-1px); }
  .dz-icon { font-size: 32px; color: var(--accent); margin-bottom: 8px; }
  .dz-title { font-weight: 600; font-size: 18px; }
  .dz-sub { color: var(--muted-2); margin-top: 6px; font-size: 14px; }
  .or { color: var(--muted-2); margin: 22px 0 12px; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
  .url-row { display: flex; gap: 10px; max-width: 620px; margin: 0 auto; }
  .url-row input { flex: 1; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text); padding: 12px 14px; border-radius: var(--radius); outline: none; }
  .url-row input:focus { border-color: var(--accent); box-shadow: var(--ring); }
  .chips { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 28px; }
  .chip { background: var(--surface); border: 1px solid var(--border); color: var(--muted); padding: 6px 12px; border-radius: 999px; font-size: 13px; }

  .primary-btn { background: var(--accent); color: #fff; font-weight: 600; padding: 12px 18px; border-radius: var(--radius); transition: background 0.15s ease, transform 0.05s ease; }
  .primary-btn:hover { background: var(--accent-hover); }
  .primary-btn:active { transform: translateY(1px); }
  .primary-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .primary-btn.big { width: 100%; padding: 14px; font-size: 16px; margin-top: 14px; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 40px; text-align: center; max-width: 560px; margin: 40px auto; box-shadow: var(--shadow); }
  .processing h2 { margin: 6px 0 4px; }
  .proc-msg { color: var(--muted); min-height: 22px; }
  .spinner { width: 46px; height: 46px; margin: 0 auto 12px; border-radius: 50%; border: 4px solid var(--border); border-top-color: var(--accent); animation: spin 0.8s linear infinite; }
  .spinner.sm { width: 30px; height: 30px; border-width: 3px; margin: 0; }
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

  .result { display: grid; grid-template-columns: 1.05fr 1fr; gap: 22px; margin-top: 24px; align-items: start; }
  @media (max-width: 860px) { .result { grid-template-columns: 1fr; } }
  .preview-panel { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 18px; box-shadow: var(--shadow); }
  .side-panel { display: flex; flex-direction: column; gap: 14px; }
  .card-sm, .howto { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px; }
  .card-sm h3, .howto h3 { margin: 0 0 12px; font-size: 14px; }

  .seg { display: inline-flex; background: var(--bg-elevated); border-radius: var(--radius); padding: 4px; gap: 4px; margin-bottom: 12px; }
  .seg.small, .seg.three { display: flex; width: 100%; }
  .seg button { flex: 1; padding: 7px 14px; border-radius: var(--radius-sm); color: var(--muted); font-weight: 600; font-size: 13px; white-space: nowrap; }
  .seg button.on { background: var(--accent); color: #fff; }

  .stage { position: relative; border-radius: var(--radius); display: grid; place-items: center; padding: 26px; min-height: 280px; }
  .stage.dark { background: #313338; }
  .stage.light { background: #ffffff; }
  .stage.checker { background-color: #2b2d31; background-image: linear-gradient(45deg, #3a3c42 25%, transparent 25%), linear-gradient(-45deg, #3a3c42 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #3a3c42 75%), linear-gradient(-45deg, transparent 75%, #3a3c42 75%); background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0; }
  .sticker { width: 220px; height: 220px; object-fit: contain; }
  .render-hint { position: absolute; bottom: 8px; right: 12px; font-size: 11px; color: var(--muted-2); background: rgba(0,0,0,0.35); padding: 2px 7px; border-radius: 6px; }
  .stage.light .render-hint { color: #6b7280; background: rgba(255,255,255,0.6); }
  .busy { position: absolute; inset: 0; display: grid; place-items: center; background: rgba(30,31,34,0.55); border-radius: var(--radius); }

  .status { display: flex; flex-direction: column; gap: 2px; margin-top: 14px; }
  .ready { color: var(--success); font-weight: 700; }
  .summary { color: var(--muted); font-size: 13px; }
  .fixes { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
  .chip-btn { background: var(--accent-soft); color: #c7ccff; border: 1px solid var(--accent); padding: 6px 12px; border-radius: 999px; font-size: 13px; font-weight: 600; }
  .chip-btn:hover { background: var(--accent); color: #fff; }

  .mini { margin-top: 12px; }
  .mini summary { cursor: pointer; color: var(--muted-2); font-size: 13px; list-style: none; }
  .mini summary::-webkit-details-marker { display: none; }
  .mini-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
  .mini-grid span { background: var(--bg-elevated); border: 1px solid var(--border); color: var(--muted); padding: 4px 10px; border-radius: 999px; font-size: 12px; }

  .presets { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
  .preset { display: flex; flex-direction: column; align-items: center; gap: 4px; padding: 12px 4px; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); color: var(--muted); font-size: 12px; font-weight: 600; transition: all 0.15s ease; }
  .preset:hover { border-color: var(--accent); color: var(--text); }
  .preset.on { border-color: var(--accent); background: var(--accent-soft); color: #fff; }
  .p-emoji { font-size: 20px; }

  .crop-wrap { display: grid; place-items: center; margin-bottom: 12px; }
  .muted-line { color: var(--muted-2); font-size: 13px; margin: 6px 0 0; }
  .toggle { display: flex; align-items: center; justify-content: space-between; font-weight: 600; color: var(--text); }
  .toggle input { width: 40px; height: 22px; accent-color: var(--accent); }

  .advanced summary { cursor: pointer; font-weight: 600; list-style: none; }
  .advanced summary::-webkit-details-marker { display: none; }
  .advanced summary::before { content: '▸ '; color: var(--accent); }
  .advanced[open] summary::before { content: '▾ '; }
  .controls { display: flex; flex-direction: column; gap: 11px; margin: 14px 0 2px; }
  .est { color: var(--muted-2); font-size: 12px; }
  .notes { list-style: none; margin: 10px 0 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
  .notes li { color: var(--muted); font-size: 12px; padding-left: 16px; position: relative; }
  .notes li::before { content: 'ⓘ'; position: absolute; left: 0; color: var(--accent); }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 14px; color: var(--muted); }
  .row span { white-space: nowrap; }
  .row input[type='range'] { flex: 1; max-width: 60%; accent-color: var(--accent); }
  .row select { background: var(--bg-elevated); color: var(--text); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 6px 8px; }

  .howto ol { margin: 0; padding-left: 18px; color: var(--muted); font-size: 14px; display: flex; flex-direction: column; gap: 6px; }
  .error .err-icon { width: 48px; height: 48px; border-radius: 50%; background: var(--danger); color: #fff; display: grid; place-items: center; font-size: 26px; font-weight: 800; margin: 0 auto 10px; }
  .error .ref { color: var(--muted-2); font-size: 13px; }
  .error code { background: var(--bg-elevated); padding: 2px 7px; border-radius: 5px; }
  .err-actions { display: flex; gap: 10px; justify-content: center; margin-top: 16px; }
  .footer { text-align: center; color: var(--muted-2); font-size: 13px; padding: 24px; }
  .drag-veil { position: fixed; inset: 0; background: rgba(30,31,34,0.82); z-index: 50; display: grid; place-items: center; backdrop-filter: blur(2px); }
  .drag-veil-inner { border: 2px dashed var(--accent); border-radius: var(--radius-lg); padding: 50px 70px; font-size: 22px; font-weight: 700; color: #fff; }
</style>
