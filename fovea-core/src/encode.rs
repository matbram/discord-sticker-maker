//! The native Fovea encode pipeline.
//!
//! Core idea (why this exists): a GIF is *not* limited to 256 colors total. Each
//! image block carries its own **local color table** of up to 256 colors, and
//! unchanged pixels can be made transparent so they cost ~0 bytes under LZW. The
//! old pipeline forced one global palette on every frame -> washout. Here every
//! frame gets its own perceptually-chosen palette (`imagequant`), and on opaque
//! content we only redraw the pixels that changed beyond a perceptual OKLab
//! threshold. That reclaimed budget is what lets us keep all frames AND rich color.
//!
//! Two modes, chosen automatically for correctness:
//!   * **delta** — fully opaque clip: `dispose=Keep`, per-frame sub-rect of changed
//!     pixels, everything else transparent (reuses the canvas). Big byte savings.
//!   * **full**  — clip has a transparent matte: independent full frames with
//!     `dispose=Background` (you cannot erase to transparent under Keep). Still gets
//!     the per-frame-palette color win; delta-with-alpha is a later refinement.

use std::borrow::Cow;
use std::collections::HashSet;

use rgb::FromSlice;

use crate::oklab::{delta_e_sq, rgb_to_oklab, srgb_to_linear_lut, Lab};

/// Tunable knobs for one encode. These map to the Python `LeverState` the search drives.
#[derive(Clone, Debug)]
pub struct EncodeOpts {
    pub width: u32,
    pub height: u32,
    /// Per-frame palette cap (2..=256). This is *per frame*, not per file.
    pub max_colors: u16,
    /// imagequant dithering level, 0.0 (none) .. 1.0 (full).
    pub dithering: f32,
    pub quality_min: u8,
    pub quality_max: u8,
    /// imagequant speed, 1 (best) .. 10 (fast).
    pub speed: i32,
    /// OKLab delta-E threshold for reusing a pixel. <= 0 disables delta (full mode).
    pub delta_threshold: f32,
    /// 0 == loop forever.
    pub loop_count: u16,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        EncodeOpts {
            width: 0,
            height: 0,
            max_colors: 256,
            dithering: 1.0,
            quality_min: 0,
            quality_max: 100,
            speed: 4,
            delta_threshold: 0.0,
            loop_count: 0,
        }
    }
}

/// Result of an encode: the GIF plus the honesty stats the report surfaces.
#[derive(Clone, Debug)]
pub struct EncodeOut {
    pub gif: Vec<u8>,
    /// Distinct (non-transparent) colors actually used per frame.
    pub colors_per_frame: Vec<u16>,
    /// Distinct colors across the whole file — the number that exceeds 256 and
    /// proves the global-palette ceiling is gone.
    pub distinct_colors: u32,
    /// Pixels carried over via delta (not re-encoded).
    pub reused_pixels: u64,
    pub total_pixels: u64,
    /// "delta" or "full".
    pub mode: &'static str,
}

/// Encode a sequence of RGBA frames (each `width*height*4` bytes) into a GIF.
pub fn encode_frames(
    frames: &[Vec<u8>],
    delays_cs: &[u16],
    opts: &EncodeOpts,
) -> Result<EncodeOut, String> {
    let w = opts.width as usize;
    let h = opts.height as usize;
    let npix = w * h;

    if frames.is_empty() {
        return Err("no frames".into());
    }
    if w == 0 || h == 0 || w > u16::MAX as usize || h > u16::MAX as usize {
        return Err(format!("bad dimensions {w}x{h}"));
    }
    if delays_cs.len() != frames.len() {
        return Err(format!(
            "delays len {} != frames {}",
            delays_cs.len(),
            frames.len()
        ));
    }
    for (i, f) in frames.iter().enumerate() {
        if f.len() != npix * 4 {
            return Err(format!("frame {i} byte len {} != {}", f.len(), npix * 4));
        }
    }

    // Delta (transparency-reuse) is only correct on fully-opaque clips: under
    // dispose=Keep you cannot turn an already-drawn pixel back to transparent, so a
    // moving alpha matte must use independent full frames.
    let has_transparency = frames
        .iter()
        .any(|f| (0..npix).any(|p| f[p * 4 + 3] < 128));
    let use_delta = opts.delta_threshold > 0.0 && !has_transparency;

    let mut liq = imagequant::new();
    liq.set_max_colors(opts.max_colors.clamp(2, 256) as u32)
        .map_err(|e| format!("imagequant max_colors: {e:?}"))?;
    liq.set_quality(opts.quality_min, opts.quality_max)
        .map_err(|e| format!("imagequant quality: {e:?}"))?;
    liq.set_speed(opts.speed.clamp(1, 10))
        .map_err(|e| format!("imagequant speed: {e:?}"))?;

    let mut out = Vec::new();
    let mut colors_per_frame: Vec<u16> = Vec::with_capacity(frames.len());
    let mut distinct: HashSet<u32> = HashSet::new();
    let mut reused_pixels: u64 = 0;
    let total_pixels: u64 = npix as u64 * frames.len() as u64;

    {
        let mut enc = gif::Encoder::new(&mut out, w as u16, h as u16, &[])
            .map_err(|e| format!("gif init: {e}"))?;
        enc.set_repeat(if opts.loop_count == 0 {
            gif::Repeat::Infinite
        } else {
            gif::Repeat::Finite(opts.loop_count)
        })
        .map_err(|e| format!("gif repeat: {e}"))?;

        if use_delta {
            encode_delta(
                &mut enc, &liq, frames, delays_cs, w, h, opts, &mut colors_per_frame,
                &mut distinct, &mut reused_pixels,
            )?;
        } else {
            encode_full(
                &mut enc, &liq, frames, delays_cs, w, h, opts, &mut colors_per_frame,
                &mut distinct,
            )?;
        }
    }

    Ok(EncodeOut {
        gif: out,
        colors_per_frame,
        distinct_colors: distinct.len() as u32,
        reused_pixels,
        total_pixels,
        mode: if use_delta { "delta" } else { "full" },
    })
}

/// Quantize an RGBA buffer to its own palette. Returns the palette, per-pixel
/// indices, and the transparent index (if any pixel was transparent).
fn quantize(
    liq: &imagequant::Attributes,
    rgba: &[u8],
    w: usize,
    h: usize,
    dithering: f32,
) -> Result<(Vec<rgb::RGBA<u8>>, Vec<u8>, Option<u8>), String> {
    let mut img = liq
        .new_image(rgba.as_rgba(), w, h, 0.0)
        .map_err(|e| format!("imagequant new_image: {e:?}"))?;
    let mut res = liq
        .quantize(&mut img)
        .map_err(|e| format!("imagequant quantize: {e:?}"))?;
    let _ = res.set_dithering_level(dithering);
    let (palette, indices) = res
        .remapped(&mut img)
        .map_err(|e| format!("imagequant remap: {e:?}"))?;
    let mut transparent = None;
    for (i, c) in palette.iter().enumerate() {
        if c.a < 128 {
            transparent = Some(i as u8);
            break;
        }
    }
    Ok((palette, indices, transparent))
}

/// Pack an RGBA palette into the GIF crate's plain-RGB table, padded to a power-of-two
/// color count (GIF color tables must be a power of two).
fn palette_to_rgb(palette: &[rgb::RGBA<u8>]) -> Vec<u8> {
    let size = palette.len().max(2).next_power_of_two().min(256);
    let mut v = Vec::with_capacity(size * 3);
    for c in palette {
        v.push(c.r);
        v.push(c.g);
        v.push(c.b);
    }
    while v.len() < size * 3 {
        v.push(0);
    }
    v
}

/// Record a frame's distinct (non-transparent) colors into the running stats.
fn record_palette(
    palette: &[rgb::RGBA<u8>],
    transparent: Option<u8>,
    colors_per_frame: &mut Vec<u16>,
    distinct: &mut HashSet<u32>,
) {
    let mut n = 0u16;
    for (i, c) in palette.iter().enumerate() {
        if Some(i as u8) == transparent {
            continue;
        }
        n += 1;
        distinct.insert((c.r as u32) << 16 | (c.g as u32) << 8 | c.b as u32);
    }
    colors_per_frame.push(n);
}

/// Independent full frames, each with its own local palette (handles any alpha).
fn encode_full<W: std::io::Write>(
    enc: &mut gif::Encoder<W>,
    liq: &imagequant::Attributes,
    frames: &[Vec<u8>],
    delays_cs: &[u16],
    w: usize,
    h: usize,
    opts: &EncodeOpts,
    colors_per_frame: &mut Vec<u16>,
    distinct: &mut HashSet<u32>,
) -> Result<(), String> {
    let npix = w * h;
    for (i, src) in frames.iter().enumerate() {
        // Hard 1-bit alpha so imagequant yields a clean transparent entry.
        let mut rgba = src.clone();
        for p in 0..npix {
            if rgba[p * 4 + 3] < 128 {
                rgba[p * 4] = 0;
                rgba[p * 4 + 1] = 0;
                rgba[p * 4 + 2] = 0;
                rgba[p * 4 + 3] = 0;
            } else {
                rgba[p * 4 + 3] = 255;
            }
        }
        let (palette, indices, transparent) = quantize(liq, &rgba, w, h, opts.dithering)?;
        record_palette(&palette, transparent, colors_per_frame, distinct);
        let mut frame = gif::Frame::default();
        frame.width = w as u16;
        frame.height = h as u16;
        frame.delay = delays_cs[i];
        frame.dispose = gif::DisposalMethod::Background;
        frame.transparent = transparent;
        frame.palette = Some(palette_to_rgb(&palette));
        frame.buffer = Cow::Owned(indices);
        enc.write_frame(&frame)
            .map_err(|e| format!("write full frame {i}: {e}"))?;
    }
    Ok(())
}

/// Opaque clips: keep a running canvas, redraw only the perceptually-changed
/// sub-rect each frame, leave the rest transparent (reused) under dispose=Keep.
#[allow(clippy::too_many_arguments)]
fn encode_delta<W: std::io::Write>(
    enc: &mut gif::Encoder<W>,
    liq: &imagequant::Attributes,
    frames: &[Vec<u8>],
    delays_cs: &[u16],
    w: usize,
    h: usize,
    opts: &EncodeOpts,
    colors_per_frame: &mut Vec<u16>,
    distinct: &mut HashSet<u32>,
    reused_pixels: &mut u64,
) -> Result<(), String> {
    let npix = w * h;
    let lut = srgb_to_linear_lut();
    let thr2 = opts.delta_threshold * opts.delta_threshold;

    // Canvas = currently displayed image (dispose=Keep persists it across frames).
    // We compare each source pixel against the *displayed* color so accumulated
    // drift can never exceed the threshold.
    let mut canvas_lab = vec![Lab::default(); npix];

    // ---- Frame 0: full, opaque ----
    {
        let mut rgba = frames[0].clone();
        for p in 0..npix {
            rgba[p * 4 + 3] = 255;
        }
        let (palette, indices, _t) = quantize(liq, &rgba, w, h, opts.dithering)?;
        record_palette(&palette, None, colors_per_frame, distinct);
        for p in 0..npix {
            let c = palette[indices[p] as usize];
            canvas_lab[p] = rgb_to_oklab(c.r, c.g, c.b, &lut);
        }
        let mut frame = gif::Frame::default();
        frame.width = w as u16;
        frame.height = h as u16;
        frame.delay = delays_cs[0];
        frame.dispose = gif::DisposalMethod::Keep;
        frame.palette = Some(palette_to_rgb(&palette));
        frame.buffer = Cow::Owned(indices);
        enc.write_frame(&frame)
            .map_err(|e| format!("write frame 0: {e}"))?;
    }

    // ---- Frames 1.. : delta sub-rects ----
    for i in 1..frames.len() {
        let src = &frames[i];
        let mut changed = vec![false; npix];
        let mut changed_count = 0usize;
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (w, h, 0usize, 0usize);
        for y in 0..h {
            for x in 0..w {
                let p = y * w + x;
                let lab = rgb_to_oklab(src[p * 4], src[p * 4 + 1], src[p * 4 + 2], &lut);
                if delta_e_sq(&lab, &canvas_lab[p]) >= thr2 {
                    changed[p] = true;
                    changed_count += 1;
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
        *reused_pixels += (npix - changed_count) as u64;

        if changed_count == 0 {
            // Nothing visibly changed: a 1x1 transparent filler carries the delay.
            colors_per_frame.push(0);
            let mut frame = gif::Frame::default();
            frame.width = 1;
            frame.height = 1;
            frame.delay = delays_cs[i];
            frame.dispose = gif::DisposalMethod::Keep;
            frame.transparent = Some(0);
            frame.palette = Some(vec![0, 0, 0, 0, 0, 0]); // 2-color (pow2) table
            frame.buffer = Cow::Owned(vec![0u8]);
            enc.write_frame(&frame)
                .map_err(|e| format!("write filler {i}: {e}"))?;
            continue;
        }

        let bw = max_x - min_x + 1;
        let bh = max_y - min_y + 1;
        // Sub-rect: changed pixels -> opaque source; unchanged -> transparent (reuse).
        let mut sub = vec![0u8; bw * bh * 4];
        for yy in 0..bh {
            for xx in 0..bw {
                let gp = (min_y + yy) * w + (min_x + xx);
                let sp = (yy * bw + xx) * 4;
                if changed[gp] {
                    sub[sp] = src[gp * 4];
                    sub[sp + 1] = src[gp * 4 + 1];
                    sub[sp + 2] = src[gp * 4 + 2];
                    sub[sp + 3] = 255;
                }
            }
        }
        let (palette, indices, transparent) = quantize(liq, &sub, bw, bh, opts.dithering)?;
        record_palette(&palette, transparent, colors_per_frame, distinct);
        // Advance the canvas for the pixels we redrew (with their quantized color).
        for yy in 0..bh {
            for xx in 0..bw {
                let gp = (min_y + yy) * w + (min_x + xx);
                if changed[gp] {
                    let c = palette[indices[yy * bw + xx] as usize];
                    canvas_lab[gp] = rgb_to_oklab(c.r, c.g, c.b, &lut);
                }
            }
        }
        let mut frame = gif::Frame::default();
        frame.left = min_x as u16;
        frame.top = min_y as u16;
        frame.width = bw as u16;
        frame.height = bh as u16;
        frame.delay = delays_cs[i];
        frame.dispose = gif::DisposalMethod::Keep;
        frame.transparent = transparent;
        frame.palette = Some(palette_to_rgb(&palette));
        frame.buffer = Cow::Owned(indices);
        enc.write_frame(&frame)
            .map_err(|e| format!("write frame {i}: {e}"))?;
    }
    Ok(())
}
