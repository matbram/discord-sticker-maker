<script>
  // "Pick the moment" for video: scrub a start point (the video seeks so you see
  // it) and choose a length. Emits { start, length } -> trim_start_s/max_duration_s.
  import { createEventDispatcher } from 'svelte'

  export let src = ''
  export let start = 0
  export let length = 4

  const dispatch = createEventDispatcher()
  let videoEl
  let duration = 0

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v))

  function onMeta() {
    duration = videoEl?.duration || 0
    start = clamp(start, 0, Math.max(0, duration - 0.3))
    length = clamp(length, 0.5, Math.min(10, duration || 10))
    emit()
  }
  function emit() { dispatch('change', { start, length }) }
  function onStart() {
    start = clamp(start, 0, Math.max(0, duration - 0.3))
    if (videoEl) { try { videoEl.currentTime = start } catch (_) {} }
    emit()
  }
  function onLength() { length = clamp(length, 0.5, Math.min(10, duration || 10)); emit() }

  $: maxStart = Math.max(0.1, (duration || 10) - 0.3)
  $: maxLen = Math.min(10, duration || 10)
</script>

<div class="trim">
  <!-- svelte-ignore a11y-media-has-caption -->
  <video bind:this={videoEl} {src} on:loadedmetadata={onMeta} muted playsinline preload="metadata"></video>
  <label class="row"><span>Start {start.toFixed(1)}s</span>
    <input type="range" min="0" max={maxStart} step="0.1" bind:value={start} on:input={onStart} /></label>
  <label class="row"><span>Length {length.toFixed(1)}s</span>
    <input type="range" min="0.5" max={maxLen} step="0.5" bind:value={length} on:input={onLength} /></label>
</div>

<style>
  .trim { display: flex; flex-direction: column; gap: 10px; }
  .trim video { width: 100%; max-height: 160px; border-radius: 8px; background: #000; object-fit: contain; }
  .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; font-size: 14px; color: var(--muted); }
  .row span { white-space: nowrap; }
  .row input[type='range'] { flex: 1; max-width: 60%; accent-color: var(--accent); }
</style>
