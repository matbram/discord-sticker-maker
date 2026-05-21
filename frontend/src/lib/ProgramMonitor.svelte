<script>
  // Live "program monitor": renders the source inside the focused output's canvas
  // (aspectW:aspectH) via a CSS transform that mirrors backend crop_fit.fit_to_canvas,
  // so what you see matches the bake. Drag to pan, scroll to zoom. When the server
  // result (bakedUrl) is ready it fades in on top for the exact asset; during edits
  // the live layer shows instantly. Video loops over the trim window.
  import { createEventDispatcher } from 'svelte'
  import { computeView, cssTransformFor, dragDelta, stageSize } from './cropMath.js'

  export let src = ''
  export let isVideo = false
  export let naturalW = 0
  export let naturalH = 0
  export let aspectW = 1
  export let aspectH = 1
  export let fitMode = 'fit'
  export let padding = 0.06
  export let zoom = 1
  export let offsetX = 0
  export let offsetY = 0
  export let trimStart = 0
  export let trimLen = 0
  export let bakedUrl = ''
  export let previewBg = 'checker'
  export let busy = false
  export let interactive = true
  export let snap = false        // snap pan to the center axes (with guide lines)
  export let maxW = 360
  export let maxH = 360

  const dispatch = createEventDispatcher()
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v))
  const SNAP = 0.05              // offset units within which we snap to a center axis

  let dragging = false
  let startX = 0, startY = 0, startOX = 0, startOY = 0
  let guideV = false, guideH = false  // show center axis guide lines while snapped
  let videoEl

  $: stage = stageSize(aspectW, aspectH, maxW, maxH)
  $: view = naturalW && naturalH
    ? computeView({ naturalW, naturalH, aspectW, aspectH, fitMode, padding, zoom, offsetX, offsetY })
    : { cx: 0, cy: 0, cw: 1, ch: 1 }
  $: tf = cssTransformFor(view, stage.w, stage.h, naturalW || 1, naturalH || 1)
  $: mediaStyle = `left:${tf.left}px;top:${tf.top}px;width:${tf.width}px;height:${tf.height}px`
  $: showBaked = !!bakedUrl && !busy && !dragging

  // keep the looping video inside [trimStart, trimStart+trimLen]
  $: if (videoEl && isVideo && trimStart >= 0) seekToStart(trimStart)
  function seekToStart(t) { try { if (Math.abs(videoEl.currentTime - t) > 0.05) videoEl.currentTime = t } catch (_) {} }
  function onTimeUpdate() {
    if (!videoEl || !trimLen) return
    const end = trimStart + trimLen
    if (videoEl.currentTime >= end || videoEl.currentTime < trimStart - 0.05) {
      try { videoEl.currentTime = trimStart } catch (_) {}
    }
  }

  function down(e) {
    if (!interactive) return
    dragging = true
    startX = e.clientX; startY = e.clientY; startOX = offsetX; startOY = offsetY
    e.target.setPointerCapture?.(e.pointerId)
  }
  function move(e) {
    if (!dragging) return
    const dx = e.clientX - startX, dy = e.clientY - startY
    const d = dragDelta(dx, dy, stage.w, stage.h)
    let nx = clamp(startOX + d.dOffsetX, -1, 1)
    let ny = clamp(startOY + d.dOffsetY, -1, 1)
    guideV = guideH = false
    if (e.shiftKey) {
      // lock to the dominant drag direction — precise single-axis pan, no lines
      if (Math.abs(dx) >= Math.abs(dy)) ny = startOY
      else nx = startOX
    } else if (snap) {
      // magnet to the center axes; show a guide line on whichever axis snapped
      if (Math.abs(nx) <= SNAP) { nx = 0; guideV = true }
      if (Math.abs(ny) <= SNAP) { ny = 0; guideH = true }
    }
    offsetX = nx; offsetY = ny
    dispatch('change', { zoom, offsetX, offsetY })
  }
  function up() { dragging = false; guideV = guideH = false }
  function wheel(e) {
    if (!interactive) return
    e.preventDefault()
    zoom = clamp(zoom * (e.deltaY < 0 ? 1.1 : 0.9), 0.3, 5)
    dispatch('change', { zoom, offsetX, offsetY })
  }
</script>

<div class="mon {previewBg}" class:interactive style="width:{stage.w}px;height:{stage.h}px"
     on:pointerdown={down} on:pointermove={move} on:pointerup={up} on:pointercancel={up} on:wheel={wheel}>
  {#if src}
    {#if isVideo}
      <!-- svelte-ignore a11y-media-has-caption -->
      <video class="media" bind:this={videoEl} {src} muted loop autoplay playsinline
             style={mediaStyle} on:timeupdate={onTimeUpdate}></video>
    {:else}
      <img class="media" {src} alt="" draggable="false" style={mediaStyle} />
    {/if}
  {/if}
  {#if bakedUrl}
    <img class="baked" class:show={showBaked} src={bakedUrl} alt="" draggable="false" />
  {/if}
  {#if guideV}<div class="guide v"></div>{/if}
  {#if guideH}<div class="guide h"></div>{/if}
  {#if interactive}<div class="hint">drag to move · scroll to zoom · ⇧ lock axis</div>{/if}
</div>

<style>
  .mon { position: relative; overflow: hidden; border-radius: 10px; touch-action: none; }
  .mon.interactive { cursor: grab; }
  .mon.interactive:active { cursor: grabbing; }
  .media { position: absolute; user-select: none; pointer-events: none; }
  .baked { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; opacity: 0; transition: opacity 0.18s ease; pointer-events: none; }
  .baked.show { opacity: 1; }
  .guide { position: absolute; background: var(--accent); pointer-events: none; box-shadow: 0 0 4px var(--accent); }
  .guide.v { top: 0; bottom: 0; left: 50%; width: 2px; transform: translateX(-1px); }
  .guide.h { left: 0; right: 0; top: 50%; height: 2px; transform: translateY(-1px); }
  .hint { position: absolute; bottom: 6px; left: 0; right: 0; text-align: center; font-size: 11px; color: var(--muted); background: rgba(0,0,0,0.35); padding: 2px; pointer-events: none; }
  .checker {
    background-color: #2b2d31;
    background-image:
      linear-gradient(45deg, #3a3c42 25%, transparent 25%), linear-gradient(-45deg, #3a3c42 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #3a3c42 75%), linear-gradient(-45deg, transparent 75%, #3a3c42 75%);
    background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0;
  }
  .dark { background: #313338; }
  .light { background: #fff; }
</style>
