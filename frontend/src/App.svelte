<script>
  import { onMount } from 'svelte'
  import { startProcess, subscribeEvents, resultUrl } from './lib/api.js'
  import { logger } from './lib/logger.js'
  import ProgramMonitor from './lib/ProgramMonitor.svelte'
  import OutputStrip from './lib/OutputStrip.svelte'
  import Timeline from './lib/Timeline.svelte'
  import { resolveAspect } from './lib/cropMath.js'

  const TYPES = [
    { id: 'gif', emoji: '🎞️', label: 'GIF', blurb: 'shareable in chat' },
    { id: 'sticker', emoji: '🏷️', label: 'Sticker', blurb: '320×320 chat sticker' },
    { id: 'emoji', emoji: '😄', label: 'Emoji', blurb: '128×128 custom emoji' }
  ]
  const STAGE_LABELS = { upload: 'Uploading', decode: 'Reading', bg: 'Cutting out', encode: 'Encoding', output_done: 'Encoding', done: 'Done' }

  function defaultParams() {
    return {
      remove_bg: false, bg_model: 'auto', auto_crop: true, fit_mode: 'fit',
      zoom: 1.0, offset_x: 0.0, offset_y: 0.0, padding: 0.06,
      max_fps: 18, max_duration_s: 4.0, trim_start_s: 0.0,
      priority: 'balanced', max_colors: 256, gif_quality: 'balanced', gif_aspect: 'source'
    }
  }

  let view = 'idle' // idle | choosing | working | done | error
  let params = defaultParams()
  let source = null
  let urlInput = ''
  let dragging = false

  // source preview / analysis
  let sourceUrl = ''
  let sourceIsVideo = false
  let sourceAnimated = false
  let sourceW = 0
  let sourceH = 0

  let selected = { gif: true, sticker: true, emoji: true }
  let focusedType = 'gif'
  // Per-output framing (pan/zoom/fit) — each format is independent.
  function defaultFraming() {
    const f = { zoom: 1, offset_x: 0, offset_y: 0, fit_mode: 'fit' }
    return { gif: { ...f }, sticker: { ...f }, emoji: { ...f } }
  }
  let framing = defaultFraming()

  let jobId = null
  let doneJob = null // job whose results are actually ready (drives preview URLs)
  let uploadPct = 0
  let progress = { stage: '', message: '', done: null, total: null }
  let outputs = [] // [{type, format, meta}]
  let previewBg = 'checker'
  let snapAxis = false // snap pan to center axes (guide lines)
  let error = { message: '', requestId: '' }
  let closeStream = null
  let watchdogTimer = null
  let busy = false
  let pendingRegen = false
  let regenTimer = null
  let fileInput

  const PARAMS_KEY = 'dsm_params_v3'
  onMount(() => {
    try { const s = JSON.parse(localStorage.getItem(PARAMS_KEY) || 'null'); if (s) params = { ...defaultParams(), ...s } } catch (_) {}
  })
  function persistParams() { try { localStorage.setItem(PARAMS_KEY, JSON.stringify(params)) } catch (_) {} }
  function fmtBytes(n) { return n == null ? '—' : (n < 1024 ? `${n} B` : n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`) }

  function armWatchdog() {
    clearTimeout(watchdogTimer)
    watchdogTimer = setTimeout(() => {
      if (busy) { if (closeStream) closeStream(); busy = false; view = 'error'
        error = { message: 'This is taking longer than expected — the server may have restarted. Please try again.', requestId: '' }
        logger.error('watchdog_timeout') }
    }, 120000)
  }
  function disarmWatchdog() { clearTimeout(watchdogTimer); watchdogTimer = null }

  function clearSourceUrl() { if (sourceUrl && sourceUrl.startsWith('blob:')) try { URL.revokeObjectURL(sourceUrl) } catch (_) {} }
  function loadDims(url, isVideo) {
    sourceW = 0; sourceH = 0
    if (isVideo) { const v = document.createElement('video'); v.onloadedmetadata = () => { sourceW = v.videoWidth; sourceH = v.videoHeight }; v.src = url }
    else { const im = new Image(); im.onload = () => { sourceW = im.naturalWidth; sourceH = im.naturalHeight }; im.src = url }
  }

  function analyze() {
    loadDims(sourceUrl, sourceIsVideo)
    sourceAnimated = sourceIsVideo || /\.gif(\?|$)/i.test(source.file?.name || source.url || '') || (source.file?.type === 'image/gif')
    // make everything by default — the editor lets them toggle outputs off
    selected = { gif: true, sticker: true, emoji: true }
  }

  function pickFiles(list) {
    const file = [...(list || [])].find((f) => f && f.size > 0)
    if (!file) return
    clearSourceUrl()
    source = { file }
    sourceIsVideo = (file.type || '').startsWith('video/')
    sourceUrl = URL.createObjectURL(file)
    params = { ...params, trim_start_s: 0, max_duration_s: 4.0 }
    framing = defaultFraming()
    analyze()
    create()
  }
  function submitUrl() {
    const url = urlInput.trim(); if (!url) return
    clearSourceUrl()
    source = { url }
    sourceIsVideo = /\.(mp4|mov|webm|mkv|avi)(\?|$)/i.test(url)
    sourceUrl = url
    params = { ...params, trim_start_s: 0, max_duration_s: 4.0 }
    framing = defaultFraming()
    analyze()
    create()
  }

  function buildOutputs() {
    const f = (t) => ({ zoom: framing[t].zoom, offset_x: framing[t].offset_x, offset_y: framing[t].offset_y, fit_mode: framing[t].fit_mode })
    const list = []
    if (selected.gif) list.push({ type: 'gif', gif_quality: params.gif_quality, aspect: params.gif_aspect, ...f('gif') })
    if (selected.sticker) list.push({ type: 'sticker', priority: params.priority, ...f('sticker') })
    if (selected.emoji) list.push({ type: 'emoji', priority: params.priority, ...f('emoji') })
    return list
  }
  $: anySelected = selected.sticker || selected.emoji || selected.gif
  function firstSelected() { return TYPES.find((t) => selected[t.id])?.id || 'sticker' }
  $: if (!selected[focusedType]) focusedType = firstSelected()

  function create() { if (!(selected.sticker || selected.emoji || selected.gif)) return; focusedType = firstSelected(); view = 'working'; run() }

  async function run() {
    if (!source || !anySelected) return
    if (closeStream) closeStream()
    persistParams()
    busy = true; uploadPct = 0
    if (view !== 'done') { view = 'working'; progress = { stage: 'upload', message: 'Uploading…', done: null, total: null } }
    error = { message: '', requestId: '' }
    try {
      const payload = { ...params, outputs: buildOutputs() }
      const { job_id } = await startProcess(source, payload, (p) => { uploadPct = p })
      jobId = job_id
      closeStream = subscribeEvents(job_id, onEvent)
      armWatchdog()
    } catch (e) {
      busy = false; disarmWatchdog(); view = 'error'
      error = { message: e.message || 'Something went wrong', requestId: e.requestId || '' }
      logger.error('run_failed', { error: String(e) })
    }
  }

  function scheduleRegen() { persistParams(); clearTimeout(regenTimer); regenTimer = setTimeout(() => { if (busy) pendingRegen = true; else run() }, 500) }

  function onEvent(evt) {
    armWatchdog()
    if (evt.type === 'progress') {
      progress = { stage: evt.stage, message: evt.message, done: evt.done, total: evt.total }
    } else if (evt.type === 'result') {
      disarmWatchdog(); busy = false
      outputs = evt.outputs || []
      doneJob = jobId
      view = 'done'
      logger.event('done', { outputs: outputs.map((o) => o.type) })
      if (pendingRegen) { pendingRegen = false; run() }
    } else if (evt.type === 'error') {
      disarmWatchdog(); busy = false; view = 'error'
      error = { message: evt.error || 'Processing failed', requestId: evt.request_id || '' }
      logger.error('failed', { error: evt.error })
    }
  }

  function toggleOutput(id) {
    selected = { ...selected, [id]: !selected[id] }
    if (!anySelected) { selected = { ...selected, [id]: true }; return } // keep at least one
    if (view === 'done') scheduleRegen()
  }
  // Sticker & emote share one locked framing (both are square crops); gif is independent.
  const LINKED = ['sticker', 'emoji']
  function applyFraming(patch) {
    const keys = LINKED.includes(focusedType) ? LINKED : [focusedType]
    const next = { ...framing }
    for (const k of keys) next[k] = { ...framing[k], ...patch }
    framing = next
  }
  function onCropChange(e) { applyFraming({ zoom: e.detail.zoom, offset_x: e.detail.offsetX, offset_y: e.detail.offsetY }); scheduleRegen() }
  function setFit(m) { applyFraming({ zoom: 1, offset_x: 0, offset_y: 0, fit_mode: m }); scheduleRegen() }
  function setPriority(p) { params = { ...params, priority: p }; scheduleRegen() }
  function setGifQuality(q) { params = { ...params, gif_quality: q }; scheduleRegen() }
  function setGifAspect(a) { params = { ...params, gif_aspect: a }; scheduleRegen() }
  function toggleBg() { params = { ...params, remove_bg: !params.remove_bg }; scheduleRegen() }
  function onTrimChange(e) { params = { ...params, trim_start_s: e.detail.start, max_duration_s: e.detail.length }; scheduleRegen() }
  function focusOutput(e) { focusedType = e.detail.type }

  function reset() {
    if (closeStream) closeStream(); disarmWatchdog(); clearSourceUrl()
    busy = false; view = 'idle'; source = null; urlInput = ''; sourceUrl = ''; sourceW = 0; sourceH = 0
    outputs = []; jobId = null; doneJob = null
  }

  function download(type) {
    const a = document.createElement('a')
    a.href = resultUrl(doneJob, { type, download: true })
    document.body.appendChild(a); a.click(); a.remove()
    logger.event('download', { type })
  }
  function downloadAll() { download('all') }

  function previewUrl(type) { return doneJob ? `/api/result/${doneJob}/${type}?v=${doneJob}` : '' }
  function getOut(type) { return outputs.find((o) => o.type === type) }

  $: fr = framing[focusedType] || framing.gif
  $: focusAspect = focusedType === 'gif' ? resolveAspect(params.gif_aspect, sourceW, sourceH) : [1, 1]
  $: focusOut = outputs.find((o) => o.type === focusedType)
  $: focusBaked = focusOut && doneJob ? previewUrl(focusedType) : ''
  $: focusMeta = TYPES.find((t) => t.id === focusedType) || TYPES[0]

  // ---- global drag & paste ----
  function onDragOver(e) { e.preventDefault(); dragging = true }
  function onDragLeave(e) { if (e.relatedTarget === null) dragging = false }
  function onDrop(e) { e.preventDefault(); dragging = false
    if (e.dataTransfer?.files?.length) pickFiles(e.dataTransfer.files)
    else { const t = e.dataTransfer?.getData('text'); if (t && /^https?:\/\//i.test(t)) { urlInput = t; submitUrl() } } }
  function onPaste(e) {
    if (busy) return
    for (const it of e.clipboardData?.items || []) { if (it.type?.startsWith('image/')) { pickFiles([it.getAsFile()]); return } }
    const t = e.clipboardData?.getData('text'); if (t && /^https?:\/\//i.test(t.trim())) { urlInput = t.trim(); submitUrl() }
  }

  $: overallPct = uploadPct < 1 ? Math.round(uploadPct * 100) : (progress.total ? Math.round((progress.done / progress.total) * 100) : null)
  $: stageText = uploadPct < 1 ? `Uploading ${Math.round(uploadPct * 100)}%` : (progress.message || 'Working…')
  $: guidance = {
    sticker: ['Server Settings → Stickers → Upload', 'Needs the Manage Expressions permission', 'Give it a name + related emoji'],
    emoji: ['Server Settings → Emoji → Upload Emoji', 'Animated emoji need a boosted server slot', 'Pick a short name (e.g. :catvibe:)'],
    gif: ['Just drag it into any chat', 'Small GIFs autoplay inline', 'Or attach it like any file']
  }
</script>

<svelte:window on:dragover={onDragOver} on:dragleave={onDragLeave} on:drop={onDrop} on:paste={onPaste} />

{#if dragging}<div class="drag-veil"><div class="drag-veil-inner">Drop to start</div></div>{/if}

<header class="topbar">
  <div class="brand"><span class="brand-mark">◈</span><span class="brand-name">Discord Media Studio</span></div>
  {#if view !== 'idle'}<button class="ghost-btn" on:click={reset}>＋ New</button>{/if}
</header>

<main class="wrap">
  {#if view === 'idle'}
    <section class="hero">
      <h1>Make Discord <span class="grad">stickers, emoji & GIFs</span></h1>
      <p class="sub">Drop any image, GIF, or video once — turn it into whatever you need, perfectly sized for Discord.</p>
      <button class="dropzone" class:drag={dragging} on:click={() => fileInput.click()} aria-label="Upload">
        <div class="dz-icon">⬆</div><div class="dz-title">Drop a file, click to browse, or paste</div>
        <div class="dz-sub">Images · GIFs · Video — any format</div>
      </button>
      <input bind:this={fileInput} type="file" accept="image/*,video/*" hidden on:change={(e) => pickFiles(e.target.files)} />
      <div class="or">or paste a link</div>
      <form class="url-row" on:submit|preventDefault={submitUrl}>
        <input type="url" placeholder="https://…/clip.mp4" bind:value={urlInput} />
        <button type="submit" class="primary-btn" disabled={!urlInput.trim()}>Continue</button>
      </form>
    </section>

  {:else if view === 'working'}
    <section class="card processing">
      <div class="spinner" aria-hidden="true"></div>
      <h2>Making your {Object.keys(selected).filter((k) => selected[k]).join(' · ')}…</h2>
      <p class="proc-msg">{stageText}</p>
      <div class="bar" class:indeterminate={overallPct == null}><div class="bar-fill" style={overallPct != null ? `width:${overallPct}%` : ''}></div></div>
    </section>

  {:else if view === 'done'}
    <section class="editor">
      <!-- main: program monitor + timeline + live output strip -->
      <div class="stage-col">
        <div class="monitor-card">
          <div class="monitor-head">
            <span class="t-emoji mh-emoji">{focusMeta.emoji}</span><b>{focusMeta.label}</b>
            {#if focusOut}<span class="out-meta">{focusOut.meta.width}×{focusOut.meta.height} · {focusOut.format} · {fmtBytes(focusOut.meta.bytes)}{focusOut.meta.animated ? ` · ${focusOut.meta.frames}f` : ''}</span>{/if}
          </div>
          {#if sourceUrl && sourceW > 0}
            <div class="monitor-wrap">
              <ProgramMonitor src={sourceUrl} isVideo={sourceIsVideo} naturalW={sourceW} naturalH={sourceH}
                              aspectW={focusAspect[0]} aspectH={focusAspect[1]}
                              fitMode={fr.fit_mode} padding={params.padding}
                              zoom={fr.zoom} offsetX={fr.offset_x} offsetY={fr.offset_y}
                              trimStart={params.trim_start_s} trimLen={params.max_duration_s}
                              bakedUrl={focusBaked} {previewBg} {busy} snap={snapAxis} maxW={400} maxH={400}
                              on:change={onCropChange} />
            </div>
          {/if}
          {#if sourceAnimated}
            <div class="transport">
              <Timeline src={sourceUrl} isVideo={sourceIsVideo}
                        start={params.trim_start_s} length={params.max_duration_s} on:change={onTrimChange} />
            </div>
          {/if}
          {#if busy}
            <div class="working-strip" aria-live="polite">
              <span class="dot-spin" aria-hidden="true"></span>
              <span class="ws-text">{stageText}</span>
              {#if overallPct != null}<span class="ws-pct">{overallPct}%</span>{/if}
              <div class="ws-bar" class:indeterminate={overallPct == null}><div class="ws-fill" style={overallPct != null ? `width:${overallPct}%` : ''}></div></div>
            </div>
          {/if}
        </div>

        <div class="strip-card">
          <div class="strip-head"><span>Your outputs</span><span class="muted-line">click a format to edit it · shared edits apply to all</span></div>
          <OutputStrip types={TYPES} {selected} {focusedType} {params} {framing}
                       src={sourceUrl} isVideo={sourceIsVideo} naturalW={sourceW} naturalH={sourceH}
                       {outputs} jobId={doneJob} {previewBg} {busy} on:focus={focusOutput} />
        </div>
      </div>

      <!-- sidebar: shared edits + focused controls + downloads -->
      <aside class="side">
        <h3>Edit</h3>
        <div class="ctl"><span class="ctl-label">Framing · {LINKED.includes(focusedType) ? 'Sticker + Emote (linked)' : focusMeta.label}</span>
          <div class="seg small">
            <button class:on={fr.fit_mode === 'fit'} on:click={() => setFit('fit')}>Fit</button>
            <button class:on={fr.fit_mode === 'fill'} on:click={() => setFit('fill')}>Fill</button>
          </div>
        </div>
        <label class="toggle"><span>Snap pan to center</span><input type="checkbox" checked={snapAxis} on:change={() => (snapAxis = !snapAxis)} /></label>
        <p class="muted-line">Snap shows center guides; hold ⇧ while dragging to lock to one axis.</p>
        <label class="toggle"><span>Cut out background</span><input type="checkbox" checked={params.remove_bg} on:change={toggleBg} /></label>

        {#if focusedType === 'gif'}
          <div class="ctl"><span class="ctl-label">GIF size</span>
            <div class="seg three">
              {#each ['small','balanced','high'] as q}<button class:on={params.gif_quality === q} on:click={() => setGifQuality(q)}>{q[0].toUpperCase() + q.slice(1)}</button>{/each}
            </div>
          </div>
          <div class="ctl"><span class="ctl-label">GIF shape</span>
            <div class="seg three">
              <button class:on={params.gif_aspect === 'square'} on:click={() => setGifAspect('square')}>Square</button>
              <button class:on={params.gif_aspect === 'source'} on:click={() => setGifAspect('source')}>Source</button>
              <button class:on={params.gif_aspect === '16:9'} on:click={() => setGifAspect('16:9')}>16:9</button>
            </div>
          </div>
        {:else if focusOut?.meta.animated}
          <div class="ctl"><span class="ctl-label">Motion</span>
            <div class="seg three">
              <button class:on={params.priority === 'smooth'} on:click={() => setPriority('smooth')}>More frames</button>
              <button class:on={params.priority === 'balanced'} on:click={() => setPriority('balanced')}>Balanced</button>
              <button class:on={params.priority === 'sharp'} on:click={() => setPriority('sharp')}>Richer</button>
            </div>
          </div>
        {/if}

        <div class="ctl"><span class="ctl-label">Preview on</span>
          <div class="seg small">
            <button class:on={previewBg === 'checker'} on:click={() => (previewBg = 'checker')}>Trans.</button>
            <button class:on={previewBg === 'dark'} on:click={() => (previewBg = 'dark')}>Dark</button>
            <button class:on={previewBg === 'light'} on:click={() => (previewBg = 'light')}>Light</button>
          </div>
        </div>

        <div class="ctl"><span class="ctl-label">Make</span>
          <div class="make-row">
            {#each TYPES as t}
              <button class="mini-toggle" class:on={selected[t.id]} on:click={() => toggleOutput(t.id)}>{t.emoji} {t.label}{selected[t.id] ? ' ✓' : ' +'}</button>
            {/each}
          </div>
        </div>

        {#if focusOut}
          <ul class="checks">
            {#each Object.entries(focusOut.meta.checklist) as [label, ok]}<li class:ok>{ok ? '✓' : '✕'} {label}</li>{/each}
          </ul>
          {#if focusOut.meta.notes?.length}<div class="onote">{focusOut.meta.notes[0]}</div>{/if}
        {/if}

        <button class="primary-btn big" on:click={() => download(focusedType)} disabled={busy || !focusOut}>⬇ Download {focusMeta.label}</button>
        {#if outputs.length > 1}<button class="ghost-btn full" on:click={downloadAll} disabled={busy}>⬇ Download all ({outputs.length})</button>{/if}

        <details class="howto-mini"><summary>How to add {focusMeta.label} to Discord</summary>
          <ol>{#each guidance[focusedType] as step}<li>{step}</li>{/each}</ol></details>
      </aside>
    </section>

  {:else if view === 'error'}
    <section class="card error">
      <div class="err-icon">!</div><h2>That didn't work</h2><p>{error.message}</p>
      {#if error.requestId}<p class="ref">Reference: <code>{error.requestId}</code></p>{/if}
      <div class="err-actions">
        <button class="primary-btn" on:click={reset}>Try another file</button>
        {#if source}<button class="ghost-btn" on:click={() => { view = 'working'; run() }}>Retry</button>{/if}
      </div>
    </section>
  {/if}
</main>

<footer class="footer">Self-hosted · no tracking · your files are processed then discarded</footer>

<style>
  .topbar { display: flex; align-items: center; justify-content: space-between; padding: 18px 24px; max-width: 1100px; margin: 0 auto; }
  .brand { display: flex; align-items: center; gap: 10px; font-weight: 700; }
  .brand-mark { color: var(--accent); font-size: 22px; }
  .ghost-btn { color: var(--muted); padding: 8px 14px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface); font-weight: 600; }
  .ghost-btn:hover { color: var(--text); border-color: var(--accent); }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 12px 24px 60px; }

  .hero { text-align: center; padding: 36px 0 0; }
  .hero h1 { font-size: clamp(28px, 5vw, 48px); font-weight: 800; margin: 0 0 12px; letter-spacing: -1px; }
  .grad { background: linear-gradient(90deg, #5865f2, #8b5cf6); -webkit-background-clip: text; background-clip: text; color: transparent; }
  .sub { color: var(--muted); max-width: 560px; margin: 0 auto 26px; font-size: 16px; }
  .dropzone { display: block; width: 100%; max-width: 620px; margin: 0 auto; background: var(--surface); border: 2px dashed var(--border); border-radius: var(--radius-lg); padding: 44px 24px; text-align: center; transition: all 0.18s ease; }
  .dropzone:hover, .dropzone.drag { border-color: var(--accent); background: var(--accent-soft); }
  .dz-icon { font-size: 32px; color: var(--accent); margin-bottom: 8px; }
  .dz-title { font-weight: 600; font-size: 18px; }
  .dz-sub { color: var(--muted-2); margin-top: 6px; font-size: 14px; }
  .or { color: var(--muted-2); margin: 22px 0 12px; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
  .url-row { display: flex; gap: 10px; max-width: 620px; margin: 0 auto; }
  .url-row input { flex: 1; background: var(--bg-elevated); border: 1px solid var(--border); color: var(--text); padding: 12px 14px; border-radius: var(--radius); outline: none; }
  .url-row input:focus { border-color: var(--accent); box-shadow: var(--ring); }

  .primary-btn { background: var(--accent); color: #fff; font-weight: 600; padding: 11px 18px; border-radius: var(--radius); transition: background 0.15s ease; }
  .primary-btn:hover { background: var(--accent-hover); }
  .primary-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .primary-btn.big { width: 100%; padding: 13px; font-size: 15px; }

  .t-emoji { font-size: 30px; }

  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 40px; text-align: center; max-width: 540px; margin: 40px auto; box-shadow: var(--shadow); }
  .processing h2 { margin: 6px 0 4px; font-size: 20px; }
  .proc-msg { color: var(--muted); min-height: 22px; }
  .spinner { width: 46px; height: 46px; margin: 0 auto 12px; border-radius: 50%; border: 4px solid var(--border); border-top-color: var(--accent); animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .bar { height: 8px; background: var(--bg-elevated); border-radius: 999px; overflow: hidden; margin-top: 8px; }
  .bar-fill { height: 100%; background: linear-gradient(90deg, #5865f2, #8b5cf6); border-radius: 999px; transition: width 0.25s ease; }
  .bar.indeterminate .bar-fill { width: 35%; animation: slide 1.1s ease-in-out infinite; }
  @keyframes slide { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }

  .editor { display: grid; grid-template-columns: 1fr 320px; gap: 22px; margin-top: 24px; align-items: start; }
  @media (max-width: 900px) { .editor { grid-template-columns: 1fr; } }
  .stage-col { display: flex; flex-direction: column; gap: 16px; min-width: 0; }
  .monitor-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px; display: flex; flex-direction: column; gap: 12px; align-items: center; }
  .monitor-head { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; width: 100%; }
  .mh-emoji { font-size: 18px; }
  .monitor-wrap { display: grid; place-items: center; }
  .transport { width: 100%; max-width: 460px; }
  .strip-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .strip-head { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; flex-wrap: wrap; font-weight: 700; font-size: 14px; }
  .side { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-lg); padding: 18px; display: flex; flex-direction: column; gap: 12px; position: sticky; top: 16px; }
  .side h3 { margin: 0; font-size: 15px; }
  .ctl { display: flex; flex-direction: column; gap: 6px; }
  .ctl-label { font-size: 12px; font-weight: 600; color: var(--muted); }
  .muted-line { color: var(--muted-2); font-size: 12px; margin: 0; font-weight: 400; }
  .toggle { display: flex; align-items: center; justify-content: space-between; font-weight: 600; }
  .toggle input { width: 38px; height: 22px; accent-color: var(--accent); }
  .make-row { display: flex; flex-wrap: wrap; gap: 6px; }
  .mini-toggle { flex: 1; min-width: 80px; padding: 8px; border-radius: var(--radius-sm); border: 1px solid var(--border); background: var(--bg-elevated); color: var(--muted); font-size: 13px; font-weight: 600; }
  .mini-toggle.on { border-color: var(--accent); color: #fff; background: var(--accent-soft); }

  .seg { display: flex; background: var(--bg-elevated); border-radius: var(--radius); padding: 4px; gap: 4px; }
  .seg button { flex: 1; padding: 6px 8px; border-radius: var(--radius-sm); color: var(--muted); font-weight: 600; font-size: 12px; }
  .seg button.on { background: var(--accent); color: #fff; }
  .ghost-btn.full { width: 100%; text-align: center; }

  .working-strip { width: 100%; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; background: var(--surface); border: 1px solid var(--accent); border-radius: var(--radius); padding: 10px 14px; }
  .dot-spin { width: 16px; height: 16px; border-radius: 50%; border: 2px solid var(--border); border-top-color: var(--accent); animation: spin 0.8s linear infinite; flex: none; }
  .ws-text { color: var(--text); font-size: 13px; font-weight: 600; }
  .ws-pct { color: var(--accent); font-size: 13px; font-weight: 700; margin-left: auto; }
  .ws-bar { flex-basis: 100%; height: 6px; background: var(--bg-elevated); border-radius: 999px; overflow: hidden; }
  .ws-fill { height: 100%; background: linear-gradient(90deg, #5865f2, #8b5cf6); border-radius: 999px; transition: width 0.25s ease; }
  .ws-bar.indeterminate .ws-fill { width: 35%; animation: slide 1.1s ease-in-out infinite; }
  .out-meta { color: var(--muted-2); font-size: 11px; margin-left: auto; }
  .checks { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 5px; }
  .checks li { color: var(--muted); font-size: 12px; }
  .checks li.ok { color: var(--text); }
  .onote { color: var(--muted); font-size: 12px; background: var(--bg-elevated); border-radius: var(--radius-sm); padding: 8px; }
  .howto-mini summary { cursor: pointer; color: var(--muted-2); font-size: 12px; }
  .howto-mini ol { margin: 8px 0 0; padding-left: 16px; color: var(--muted); font-size: 12px; display: flex; flex-direction: column; gap: 4px; }

  .error .err-icon { width: 48px; height: 48px; border-radius: 50%; background: var(--danger); color: #fff; display: grid; place-items: center; font-size: 26px; font-weight: 800; margin: 0 auto 10px; }
  .error .ref { color: var(--muted-2); font-size: 13px; } .error code { background: var(--bg-elevated); padding: 2px 7px; border-radius: 5px; }
  .err-actions { display: flex; gap: 10px; justify-content: center; margin-top: 16px; }
  .footer { text-align: center; color: var(--muted-2); font-size: 13px; padding: 24px; }
  .drag-veil { position: fixed; inset: 0; background: rgba(30,31,34,0.82); z-index: 50; display: grid; place-items: center; backdrop-filter: blur(2px); }
  .drag-veil-inner { border: 2px dashed var(--accent); border-radius: var(--radius-lg); padding: 50px 70px; font-size: 22px; font-weight: 700; color: #fff; }
</style>
