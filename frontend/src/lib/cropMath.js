// Client-side mirror of backend crop_fit.fit_to_canvas. Keep these formulae in
// lockstep with backend/app/pipeline/crop_fit.py so the live preview matches the
// baked asset. Previews use the full frame as the bbox (the alpha subject-bbox is
// server-only); with background-removal on, the baked overlay corrects the small
// difference once it arrives.

// Crop rectangle (in source pixels) for a target aspect + framing.
export function computeView({
  naturalW, naturalH, aspectW = 1, aspectH = 1,
  fitMode = 'fit', padding = 0.06, zoom = 1, offsetX = 0, offsetY = 0, bbox = null,
}) {
  const ar = aspectW / aspectH
  let [x0, y0, x1, y1] = bbox || [0, 0, naturalW, naturalH]

  let cw
  if (fitMode === 'fill') {
    cw = Math.min(x1 - x0, (y1 - y0) * ar)
  } else {
    const pad = Math.round(Math.max(x1 - x0, y1 - y0) * padding)
    x0 -= pad; y0 -= pad; x1 += pad; y1 += pad
    cw = Math.max(x1 - x0, (y1 - y0) * ar)
  }
  cw = cw / Math.max(zoom, 1e-3)
  const ch = cw / ar

  let cx = (x0 + x1) / 2
  let cy = (y0 + y1) / 2
  cx += offsetX * cw / 2
  cy += offsetY * ch / 2
  return { cx, cy, cw, ch }
}

// Absolute position/size (px) for the source <img>/<video> inside a stageW x stageH
// viewport so that the crop rect fills the stage. Stage aspect should equal aspectW:aspectH.
export function cssTransformFor(view, stageW, stageH, naturalW, naturalH) {
  const scale = stageW / view.cw
  return {
    width: naturalW * scale,
    height: naturalH * scale,
    left: stageW / 2 - view.cx * scale,
    top: stageH / 2 - view.cy * scale,
  }
}

// Offset deltas for a drag of (dx, dy) screen px on a stageW x stageH viewport.
export function dragDelta(dx, dy, stageW, stageH) {
  return { dOffsetX: -2 * dx / stageW, dOffsetY: -2 * dy / stageH }
}

// Mirror of backend models.resolve_aspect: GIF aspect choice → [w, h] ratio.
export function resolveAspect(aspect, srcW, srcH) {
  if (aspect === 'square') return [1, 1]
  if (aspect === '16:9') return [16, 9]
  return [Math.max(1, srcW || 1), Math.max(1, srcH || 1)]
}

// Fit an output aspect into a max box, preserving aspect → on-screen stage size.
export function stageSize(aspectW, aspectH, maxW, maxH) {
  const ar = aspectW / aspectH
  let w = maxW
  let h = w / ar
  if (h > maxH) { h = maxH; w = h * ar }
  return { w: Math.round(w), h: Math.round(h) }
}
