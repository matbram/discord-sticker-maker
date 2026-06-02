<script>
  // A row of live mini-previews — one per selected output. All share the same
  // framing/trim params, so editing the program monitor updates them together.
  // Click one to focus it in the monitor. Shows the baked asset once ready.
  import { createEventDispatcher } from 'svelte'
  import ProgramMonitor from './ProgramMonitor.svelte'

  export let types = []          // [{ id, emoji, label }]
  export let selected = {}
  export let focusedType = ''
  export let params = {}
  export let framing = {}
  export let gifAR = [1, 1]      // GIF target [w, h] (from App: custom W×H or source shape)
  export let src = ''
  export let isVideo = false
  export let naturalW = 0
  export let naturalH = 0
  export let outputs = []
  export let jobId = null
  export let previewBg = 'checker'
  export let busy = false

  const dispatch = createEventDispatcher()
  const fmtBytes = (n) => n == null ? '' : n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1048576).toFixed(1)} MB`

  $: shown = types.filter((t) => selected[t.id])
  function aspectFor(id) {
    if (id === 'gif') return gifAR
    return [1, 1]
  }
  function getOut(id) { return outputs.find((o) => o.type === id) }
  function bakedFor(id) { return getOut(id) && jobId ? `/api/result/${jobId}/${id}?v=${jobId}` : '' }
</script>

<div class="strip">
  {#each shown as t (t.id)}
    {@const o = getOut(t.id)}
    {@const ar = aspectFor(t.id)}
    {@const fr = framing[t.id] || { zoom: 1, offset_x: 0, offset_y: 0, fit_mode: 'fit' }}
    <button class="mini" class:focused={focusedType === t.id} on:click={() => dispatch('focus', { type: t.id })}>
      <div class="mini-stage">
        <ProgramMonitor {src} {isVideo} {naturalW} {naturalH}
                        aspectW={ar[0]} aspectH={ar[1]}
                        fitMode={fr.fit_mode} padding={params.padding}
                        zoom={fr.zoom} offsetX={fr.offset_x} offsetY={fr.offset_y}
                        trimStart={params.trim_start_s} trimLen={params.max_duration_s}
                        bakedUrl={bakedFor(t.id)} {previewBg} {busy}
                        interactive={false} maxW={116} maxH={116} />
      </div>
      <div class="mini-label"><span class="me">{t.emoji}</span> {t.label}</div>
      {#if o}<div class="mini-meta">{o.meta.width}×{o.meta.height} · {fmtBytes(o.meta.bytes)}{o.meta.animated ? ` · ${o.meta.frames}f` : ''}</div>{/if}
      <div class="mini-edit">{focusedType === t.id ? '● Editing' : '✎ Edit'}</div>
    </button>
  {/each}
</div>

<style>
  .strip { display: flex; flex-wrap: wrap; gap: 12px; }
  .mini { display: flex; flex-direction: column; align-items: center; gap: 6px; padding: 8px; border-radius: 12px; border: 2px solid var(--border); background: var(--bg-elevated); transition: border-color 0.15s ease, transform 0.1s ease; }
  .mini:hover { border-color: var(--accent); }
  .mini.focused { border-color: var(--accent); background: var(--accent-soft); }
  .mini-stage { width: 116px; height: 116px; display: grid; place-items: center; }
  .mini-stage :global(.mon) { margin: 0 auto; }
  .mini-label { font-size: 12px; font-weight: 600; color: var(--text); }
  .mini-label .me { font-size: 14px; }
  .mini-meta { font-size: 10px; color: var(--muted-2); }
  .mini-edit { font-size: 10px; font-weight: 700; padding: 2px 9px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); }
  .mini:hover .mini-edit { border-color: var(--accent); color: var(--text); }
  .mini.focused .mini-edit { background: var(--accent); color: #fff; border-color: var(--accent); }
</style>
