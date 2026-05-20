<script>
  // Interactive "viewfinder" on the source. Drag to reposition, wheel/slider to
  // zoom. The whole stage represents the final 320x320 sticker, so what you see
  // is what you get. Math mirrors backend crop_fit.fit_frames (no-bg case).
  import { createEventDispatcher } from 'svelte'

  export let src = ''
  export let isVideo = false
  export let naturalW = 0
  export let naturalH = 0
  export let zoom = 1
  export let offsetX = 0
  export let offsetY = 0
  export let fitMode = 'fit'
  export let padding = 0.06

  const dispatch = createEventDispatcher()
  const S = 280 // on-screen stage size (px)

  let dragging = false
  let lastX = 0
  let lastY = 0

  const clamp = (v, a, b) => Math.max(a, Math.min(b, v))

  $: pad = fitMode === 'fill' ? 0 : Math.round(Math.max(naturalW, naturalH) * padding)
  $: base = naturalW && naturalH
    ? (fitMode === 'fill' ? Math.min(naturalW, naturalH) : Math.max(naturalW, naturalH) + 2 * pad)
    : 1
  $: side = base / Math.max(zoom, 0.001)
  $: scale = S / side
  $: centerX = naturalW / 2 + offsetX * side / 2
  $: centerY = naturalH / 2 + offsetY * side / 2
  $: imgStyle = `left:${S / 2 - centerX * scale}px;top:${S / 2 - centerY * scale}px;`
    + `width:${naturalW * scale}px;height:${naturalH * scale}px`

  function pt(e) { return { x: e.clientX, y: e.clientY } }
  function down(e) { dragging = true; const p = pt(e); lastX = p.x; lastY = p.y; e.target.setPointerCapture?.(e.pointerId) }
  function move(e) {
    if (!dragging) return
    const p = pt(e)
    offsetX = clamp(offsetX - 2 * (p.x - lastX) / S, -1, 1)
    offsetY = clamp(offsetY - 2 * (p.y - lastY) / S, -1, 1)
    lastX = p.x; lastY = p.y
    dispatch('change', { zoom, offsetX, offsetY })
  }
  function up() { dragging = false }
  function wheel(e) {
    e.preventDefault()
    zoom = clamp(zoom * (e.deltaY < 0 ? 1.1 : 0.9), 0.5, 4)
    dispatch('change', { zoom, offsetX, offsetY })
  }
</script>

<div class="crop" style="width:{S}px;height:{S}px"
     on:pointerdown={down} on:pointermove={move} on:pointerup={up} on:pointercancel={up} on:wheel={wheel}>
  {#if isVideo}
    <!-- svelte-ignore a11y-media-has-caption -->
    <video class="media" {src} muted loop autoplay playsinline style={imgStyle}></video>
  {:else}
    <img class="media" {src} alt="" draggable="false" style={imgStyle} />
  {/if}
  <div class="frame"></div>
  <div class="hint">drag to move · scroll to zoom</div>
</div>

<style>
  .crop {
    position: relative; overflow: hidden; border-radius: 8px; cursor: grab; touch-action: none;
    background-color: #2b2d31;
    background-image:
      linear-gradient(45deg, #3a3c42 25%, transparent 25%),
      linear-gradient(-45deg, #3a3c42 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #3a3c42 75%),
      linear-gradient(-45deg, transparent 75%, #3a3c42 75%);
    background-size: 20px 20px; background-position: 0 0, 0 10px, 10px -10px, -10px 0;
  }
  .crop:active { cursor: grabbing; }
  .media { position: absolute; user-select: none; pointer-events: none; }
  .frame { position: absolute; inset: 0; box-shadow: 0 0 0 2px var(--accent) inset; border-radius: 8px; pointer-events: none; }
  .hint { position: absolute; bottom: 6px; left: 0; right: 0; text-align: center; font-size: 11px; color: var(--muted); background: rgba(0,0,0,0.35); padding: 2px; pointer-events: none; }
</style>
