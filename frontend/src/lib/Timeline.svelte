<script>
  // Scrubbable timeline + trim handles for any animated source. Duration comes from
  // video metadata, or (for GIF/APNG/WebP) WebCodecs ImageDecoder; if neither is
  // available we fall back to coarse range sliders. Emits { start, length } ->
  // trim_start_s / max_duration_s. The program monitor reseeks off trim_start.
  import { createEventDispatcher } from 'svelte'

  export let src = ''
  export let isVideo = false
  export let start = 0
  export let length = 4

  const dispatch = createEventDispatcher()
  const MIN_LEN = 0.3
  const MAX_LEN = 30
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v))

  let duration = 0
  let videoEl
  let trackEl
  let drag = null
  let bodyGrip = 0

  function emit() { dispatch('change', { start: +start.toFixed(2), length: +length.toFixed(2) }) }

  // Video reports a reliable duration -> show the scrub track. Animated images
  // (GIF/APNG/WebP) don't expose one cheaply, so they fall back to the length slider
  // below. Trim only changes on a user action — never auto-emitted on mount.
  function onMeta() { duration = videoEl?.duration || 0 }

  // ---- track dragging (when duration known) ----
  function frac(e) { const r = trackEl.getBoundingClientRect(); return clamp((e.clientX - r.left) / r.width, 0, 1) }
  function startDrag(which, e) {
    drag = which
    if (which === 'body') bodyGrip = frac(e) * duration - start
    trackEl.setPointerCapture?.(e.pointerId)
  }
  function moveDrag(e) {
    if (!drag || !duration) return
    const t = frac(e) * duration
    if (drag === 'start') { const ns = clamp(t, 0, start + length - MIN_LEN); length = (start + length) - ns; start = ns }
    else if (drag === 'end') { length = clamp(t, start + MIN_LEN, duration) - start }
    else if (drag === 'body') { start = clamp(t - bodyGrip, 0, duration - length) }
    emit()
  }
  function endDrag() { drag = null }

  // ---- fallback range sliders (duration unknown) ----
  function onStartInput() { start = clamp(start, 0, MAX_LEN); emit() }
  function onLenInput() { length = clamp(length, MIN_LEN, MAX_LEN); emit() }

  // Display values clamp the (un-emitted) trim into the known duration so the handles
  // never overflow the track; the real start/length only change on user drag.
  $: dStart = duration > 0 ? clamp(start, 0, Math.max(0, duration - MIN_LEN)) : start
  $: dLen = duration > 0 ? clamp(length, MIN_LEN, Math.max(MIN_LEN, duration - dStart)) : length
  $: pct = duration > 0 ? { left: (dStart / duration) * 100, width: (dLen / duration) * 100 } : null
  $: end = dStart + dLen
</script>

{#if isVideo}
  <!-- svelte-ignore a11y-media-has-caption -->
  <video bind:this={videoEl} {src} on:loadedmetadata={onMeta} muted preload="metadata" style="display:none"></video>
{/if}

<div class="timeline">
  {#if duration > 0}
    <div class="track" bind:this={trackEl} on:pointermove={moveDrag} on:pointerup={endDrag} on:pointercancel={endDrag}>
      <div class="rail"></div>
      <div class="sel" style="left:{pct.left}%;width:{pct.width}%" on:pointerdown={(e) => startDrag('body', e)}>
        <span class="handle h-l" on:pointerdown|stopPropagation={(e) => startDrag('start', e)}></span>
        <span class="handle h-r" on:pointerdown|stopPropagation={(e) => startDrag('end', e)}></span>
      </div>
    </div>
    <div class="times">
      <span>{dStart.toFixed(1)}s</span>
      <span class="len">{dLen.toFixed(1)}s clip</span>
      <span>{end.toFixed(1)}s</span>
    </div>
  {:else}
    <label class="row"><span>Start {start.toFixed(1)}s</span>
      <input type="range" min="0" max={MAX_LEN - 0.5} step="0.1" bind:value={start} on:input={onStartInput} /></label>
    <label class="row"><span>Length {length.toFixed(1)}s</span>
      <input type="range" min={MIN_LEN} max={MAX_LEN} step="0.1" bind:value={length} on:input={onLenInput} /></label>
  {/if}
</div>

<style>
  .timeline { display: flex; flex-direction: column; gap: 8px; }
  .track { position: relative; height: 34px; touch-action: none; cursor: pointer; display: flex; align-items: center; }
  .rail { position: absolute; left: 0; right: 0; height: 14px; border-radius: 7px; background: var(--bg-elevated); border: 1px solid var(--border); }
  .sel { position: absolute; height: 26px; top: 4px; border-radius: 6px; background: var(--accent-soft); border: 2px solid var(--accent); cursor: grab; }
  .sel:active { cursor: grabbing; }
  .handle { position: absolute; top: 50%; transform: translateY(-50%); width: 12px; height: 30px; border-radius: 4px; background: var(--accent); cursor: ew-resize; }
  .handle.h-l { left: -6px; } .handle.h-r { right: -6px; }
  .times { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); }
  .times .len { color: var(--text); font-weight: 600; }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 14px; color: var(--muted); }
  .row span { white-space: nowrap; }
  .row input[type='range'] { flex: 1; max-width: 60%; accent-color: var(--accent); }
</style>
