//! The native Fovea encode pipeline + an in-Rust byte-target search.
//!
//! Core idea (why this exists): a GIF is *not* limited to 256 colors total. Each
//! image block carries its own **local color table** of up to 256 colors, and
//! unchanged pixels can be made transparent so they cost ~0 bytes under LZW. The
//! old pipeline forced one global palette on every frame -> washout. Here every
//! frame gets its own perceptually-chosen palette (`imagequant`), and on opaque
//! content we only redraw the pixels that changed beyond a perceptual OKLab
//! threshold.
//!
//! Two assembly modes, chosen automatically for correctness:
//!   * **delta** — fully opaque clip: `dispose=Keep`, per-frame sub-rect of changed
//!     pixels, everything else transparent (reuses the canvas). Big byte savings.
//!   * **full**  — clip has a transparent matte: independent full frames with
//!     `dispose=Background`. Also used for the *search* because its per-frame
//!     quantization parallelises (delta's canvas is a sequential dependency) and
//!     its size upper-bounds delta's, so a color count that fits in full fits in delta.
//!
//! `encode_search` does the whole byte-target color search in ONE call: it bisects
//! the color ladder with the parallel full-mode sizer, then encodes the winner in
//! the real mode. It always returns the smallest-it-can result, so it can never
//! "time out into" an over-budget file the way a per-probe Python search could.

use std::borrow::Cow;
use std::collections::HashSet;

use rayon::prelude::*;
use rgb::FromSlice;

use crate::oklab::{delta_e_sq, rgb_to_oklab, srgb_to_linear_lut, Lab};

/// Tunable knobs for one fixed-setting encode.
#[derive(Clone, Debug)]
pub struct EncodeOpts {
    pub width: u32,
    pub height: u32,
    pub max_colors: u16,
    pub dithering: f32,
    pub quality_min: u8,
    pub quality_max: u8,
    pub speed: i32,
    pub delta_threshold: f32,
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
    pub colors_per_frame: Vec<u16>,
    pub distinct_colors: u32,
    pub reused_pixels: u64,
    pub total_pixels: u64,
    pub mode: &'static str,
}

/// Knobs for the byte-target search.
#[derive(Clone, Debug)]
pub struct SearchOpts {
    pub width: u32,
    pub height: u32,
    pub target_bytes: u64,
    pub max_colors: u16,
    pub min_colors: u16,
    pub dithering: f32,
    pub quality_min: u8,
    pub quality_max: u8,
    pub speed: i32,
    pub delta_threshold: f32,
    pub loop_count: u16,
}

/// Result of the search: the chosen GIF + which per-frame color budget won.
#[derive(Clone, Debug)]
pub struct SearchOut {
    pub gif: Vec<u8>,
    pub colors: u16,
    pub colors_per_frame: Vec<u16>,
    pub distinct_colors: u32,
    pub reused_pixels: u64,
    pub total_pixels: u64,
    pub mode: &'static str,
    pub under_budget: bool,
}

/// One frame's quantization: its local palette, per-pixel indices, transparent slot.
struct FrameQuant {
    palette: Vec<rgb::RGBA<u8>>,
    indices: Vec<u8>,
    transparent: Option<u8>,
}

fn validate(frames: &[Vec<u8>], delays: &[u16], w: usize, h: usize) -> Result<(), String> {
    if frames.is_empty() {
        return Err("no frames".into());
    }
    if w == 0 || h == 0 || w > u16::MAX as usize || h > u16::MAX as usize {
        return Err(format!("bad dimensions {w}x{h}"));
    }
    if delays.len() != frames.len() {
        return Err(format!("delays len {} != frames {}", delays.len(), frames.len()));
    }
    for (i, f) in frames.iter().enumerate() {
        if f.len() != w * h * 4 {
            return Err(format!("frame {i} byte len {} != {}", f.len(), w * h * 4));
        }
    }
    Ok(())
}

fn has_transparency(frames: &[Vec<u8>], npix: usize) -> bool {
    frames.iter().any(|f| (0..npix).any(|p| f[p * 4 + 3] < 128))
}

/// Quantize one RGBA frame to its own palette (self-contained so it runs on a
/// rayon worker without sharing imagequant state).
fn quantize_owned(
    rgba: &[u8],
    w: usize,
    h: usize,
    max_colors: u16,
    quality_min: u8,
    quality_max: u8,
    speed: i32,
    dithering: f32,
) -> Result<FrameQuant, String> {
    let mut liq = imagequant::new();
    liq.set_max_colors(max_colors.clamp(2, 256) as u32)
        .map_err(|e| format!("imagequant max_colors: {e:?}"))?;
    liq.set_quality(quality_min, quality_max)
        .map_err(|e| format!("imagequant quality: {e:?}"))?;
    liq.set_speed(speed.clamp(1, 10))
        .map_err(|e| format!("imagequant speed: {e:?}"))?;
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
    let transparent = palette.iter().position(|c| c.a < 128).map(|i| i as u8);
    Ok(FrameQuant { palette, indices, transparent })
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

/// Full mode: every frame independent, its own palette, `dispose=Background`. The
/// per-frame quantization runs in parallel (no cross-frame dependency).
fn encode_full_out(frames: &[Vec<u8>], delays: &[u16], opts: &EncodeOpts) -> Result<EncodeOut, String> {
    let w = opts.width as usize;
    let h = opts.height as usize;
    let npix = w * h;

    let quants: Vec<FrameQuant> = frames
        .par_iter()
        .map(|src| {
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
            quantize_owned(
                &rgba, w, h, opts.max_colors, opts.quality_min, opts.quality_max, opts.speed,
                opts.dithering,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut out = Vec::new();
    let mut colors_per_frame = Vec::with_capacity(frames.len());
    let mut distinct: HashSet<u32> = HashSet::new();
    {
        let mut enc = gif::Encoder::new(&mut out, w as u16, h as u16, &[])
            .map_err(|e| format!("gif init: {e}"))?;
        enc.set_repeat(repeat(opts.loop_count)).map_err(|e| format!("gif repeat: {e}"))?;
        for (i, q) in quants.iter().enumerate() {
            record_palette(&q.palette, q.transparent, &mut colors_per_frame, &mut distinct);
            let mut frame = gif::Frame::default();
            frame.width = w as u16;
            frame.height = h as u16;
            frame.delay = delays[i];
            frame.dispose = gif::DisposalMethod::Background;
            frame.transparent = q.transparent;
            frame.palette = Some(palette_to_rgb(&q.palette));
            frame.buffer = Cow::Borrowed(&q.indices);
            enc.write_frame(&frame).map_err(|e| format!("write full frame {i}: {e}"))?;
        }
    }
    Ok(EncodeOut {
        gif: out,
        colors_per_frame,
        distinct_colors: distinct.len() as u32,
        reused_pixels: 0,
        total_pixels: npix as u64 * frames.len() as u64,
        mode: "full",
    })
}

fn repeat(loop_count: u16) -> gif::Repeat {
    if loop_count == 0 {
        gif::Repeat::Infinite
    } else {
        gif::Repeat::Finite(loop_count)
    }
}

/// Delta mode (opaque clips): keep a running canvas, redraw only the
/// perceptually-changed sub-rect each frame, leave the rest transparent (reused)
/// under `dispose=Keep`. Sequential by nature (each frame depends on the canvas).
fn encode_delta_out(frames: &[Vec<u8>], delays: &[u16], opts: &EncodeOpts) -> Result<EncodeOut, String> {
    let w = opts.width as usize;
    let h = opts.height as usize;
    let npix = w * h;
    let lut = srgb_to_linear_lut();
    let thr2 = opts.delta_threshold * opts.delta_threshold;

    let mut out = Vec::new();
    let mut colors_per_frame: Vec<u16> = Vec::with_capacity(frames.len());
    let mut distinct: HashSet<u32> = HashSet::new();
    let mut reused_pixels: u64 = 0;
    let mut canvas_lab = vec![Lab::default(); npix];

    {
        let mut enc = gif::Encoder::new(&mut out, w as u16, h as u16, &[])
            .map_err(|e| format!("gif init: {e}"))?;
        enc.set_repeat(repeat(opts.loop_count)).map_err(|e| format!("gif repeat: {e}"))?;

        // Frame 0: full, opaque.
        {
            let mut rgba = frames[0].clone();
            for p in 0..npix {
                rgba[p * 4 + 3] = 255;
            }
            let q = quantize_owned(
                &rgba, w, h, opts.max_colors, opts.quality_min, opts.quality_max, opts.speed,
                opts.dithering,
            )?;
            record_palette(&q.palette, None, &mut colors_per_frame, &mut distinct);
            for p in 0..npix {
                let c = q.palette[q.indices[p] as usize];
                canvas_lab[p] = rgb_to_oklab(c.r, c.g, c.b, &lut);
            }
            let mut frame = gif::Frame::default();
            frame.width = w as u16;
            frame.height = h as u16;
            frame.delay = delays[0];
            frame.dispose = gif::DisposalMethod::Keep;
            frame.palette = Some(palette_to_rgb(&q.palette));
            frame.buffer = Cow::Owned(q.indices);
            enc.write_frame(&frame).map_err(|e| format!("write frame 0: {e}"))?;
        }

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
            reused_pixels += (npix - changed_count) as u64;

            if changed_count == 0 {
                colors_per_frame.push(0);
                let mut frame = gif::Frame::default();
                frame.width = 1;
                frame.height = 1;
                frame.delay = delays[i];
                frame.dispose = gif::DisposalMethod::Keep;
                frame.transparent = Some(0);
                frame.palette = Some(vec![0, 0, 0, 0, 0, 0]);
                frame.buffer = Cow::Owned(vec![0u8]);
                enc.write_frame(&frame).map_err(|e| format!("write filler {i}: {e}"))?;
                continue;
            }

            let bw = max_x - min_x + 1;
            let bh = max_y - min_y + 1;
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
            let q = quantize_owned(
                &sub, bw, bh, opts.max_colors, opts.quality_min, opts.quality_max, opts.speed,
                opts.dithering,
            )?;
            record_palette(&q.palette, q.transparent, &mut colors_per_frame, &mut distinct);
            for yy in 0..bh {
                for xx in 0..bw {
                    let gp = (min_y + yy) * w + (min_x + xx);
                    if changed[gp] {
                        let c = q.palette[q.indices[yy * bw + xx] as usize];
                        canvas_lab[gp] = rgb_to_oklab(c.r, c.g, c.b, &lut);
                    }
                }
            }
            let mut frame = gif::Frame::default();
            frame.left = min_x as u16;
            frame.top = min_y as u16;
            frame.width = bw as u16;
            frame.height = bh as u16;
            frame.delay = delays[i];
            frame.dispose = gif::DisposalMethod::Keep;
            frame.transparent = q.transparent;
            frame.palette = Some(palette_to_rgb(&q.palette));
            frame.buffer = Cow::Owned(q.indices);
            enc.write_frame(&frame).map_err(|e| format!("write frame {i}: {e}"))?;
        }
    }
    Ok(EncodeOut {
        gif: out,
        colors_per_frame,
        distinct_colors: distinct.len() as u32,
        reused_pixels,
        total_pixels: npix as u64 * frames.len() as u64,
        mode: "delta",
    })
}

/// Encode a sequence of RGBA frames into a GIF at a fixed color budget.
pub fn encode_frames(
    frames: &[Vec<u8>],
    delays_cs: &[u16],
    opts: &EncodeOpts,
) -> Result<EncodeOut, String> {
    let w = opts.width as usize;
    let h = opts.height as usize;
    validate(frames, delays_cs, w, h)?;
    let use_delta = opts.delta_threshold > 0.0 && !has_transparency(frames, w * h);
    if use_delta {
        encode_delta_out(frames, delays_cs, opts)
    } else {
        encode_full_out(frames, delays_cs, opts)
    }
}

/// Ascending per-frame color ladder, dense in the low/mid range where banding bites.
fn color_ladder(min_colors: u16, max_colors: u16) -> Vec<u16> {
    const BASE: [u16; 19] = [
        2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224, 256,
    ];
    let lo = min_colors.clamp(2, 256);
    let hi = max_colors.clamp(2, 256).max(lo);
    let mut v: Vec<u16> = BASE.iter().copied().filter(|&c| c >= lo && c <= hi).collect();
    if v.is_empty() {
        v.push(hi);
    }
    v
}

fn search_out(colors: u16, eo: EncodeOut, target: u64) -> SearchOut {
    SearchOut {
        under_budget: eo.gif.len() as u64 <= target,
        gif: eo.gif,
        colors,
        colors_per_frame: eo.colors_per_frame,
        distinct_colors: eo.distinct_colors,
        reused_pixels: eo.reused_pixels,
        total_pixels: eo.total_pixels,
        mode: eo.mode,
    }
}

/// ONE call: find the largest per-frame palette whose GIF fits `target_bytes`, then
/// hand back that GIF. Two-stage so it is both fast and not conservative:
///   1. **Parallel full-mode bisection** — frames quantize in parallel, giving a fast,
///      guaranteed-fitting result and a lower bound on the color budget.
///   2. **Delta bisection upward** from there — delta is much smaller than full on
///      real video (static/dark regions reuse), so it climbs to a far richer palette
///      that still fits. `full` at the same colors is always kept as the safety net.
/// Runs to completion, so it can never time out into an over-budget file.
pub fn encode_search(
    frames: &[Vec<u8>],
    delays_cs: &[u16],
    sopts: &SearchOpts,
) -> Result<SearchOut, String> {
    let w = sopts.width as usize;
    let h = sopts.height as usize;
    validate(frames, delays_cs, w, h)?;
    let target = sopts.target_bytes;
    let want_delta = sopts.delta_threshold > 0.0 && !has_transparency(frames, w * h);
    let ladder = color_ladder(sopts.min_colors, sopts.max_colors);

    let opts_at = |colors: u16, delta: f32| EncodeOpts {
        width: sopts.width,
        height: sopts.height,
        max_colors: colors,
        dithering: sopts.dithering,
        quality_min: sopts.quality_min,
        quality_max: sopts.quality_max,
        speed: sopts.speed,
        delta_threshold: delta,
        loop_count: sopts.loop_count,
    };

    // ---- Stage 1: parallel full-mode bisection (fast, guaranteed fit). ----
    let (mut lo, mut hi) = (0i32, ladder.len() as i32 - 1);
    let mut full_best: Option<(usize, EncodeOut)> = None;
    while lo <= hi {
        let mid = ((lo + hi) / 2) as usize;
        let eo = encode_full_out(frames, delays_cs, &opts_at(ladder[mid], 0.0))?;
        if eo.gif.len() as u64 <= target {
            full_best = Some((mid, eo));
            lo = mid as i32 + 1;
        } else {
            hi = mid as i32 - 1;
        }
    }
    let (full_idx, full_eo) = match full_best {
        Some(x) => x,
        None => (0usize, encode_full_out(frames, delays_cs, &opts_at(ladder[0], 0.0))?),
    };

    if !want_delta {
        return Ok(search_out(ladder[full_idx], full_eo, target));
    }

    // ---- Stage 2: does delta meaningfully help? Probe it once at the chosen colors. ----
    // On full-motion clips (little reuse) delta ~= full but is *sequential* (no per-frame
    // parallelism, so slow), so we must NOT do a whole delta bisection there. We climb
    // only when one probe shows real reuse; otherwise the parallel full result stands.
    let d0 = encode_delta_out(frames, delays_cs, &opts_at(ladder[full_idx], sopts.delta_threshold))?;
    let d0_fits = d0.gif.len() as u64 <= target;
    let d0_helps = d0_fits && (d0.gif.len() as u64) * 100 < (full_eo.gif.len() as u64) * 88;
    if !d0_helps {
        let chosen = if d0_fits && d0.gif.len() < full_eo.gif.len() { d0 } else { full_eo };
        return Ok(search_out(ladder[full_idx], chosen, target));
    }

    // Reuse is real -> climb delta from full_idx upward (delta encodes are cheap here).
    let (mut lo, mut hi) = (full_idx as i32 + 1, ladder.len() as i32 - 1);
    let mut delta_best: (usize, EncodeOut) = (full_idx, d0);
    while lo <= hi {
        let mid = ((lo + hi) / 2) as usize;
        let eo = encode_delta_out(frames, delays_cs, &opts_at(ladder[mid], sopts.delta_threshold))?;
        if eo.gif.len() as u64 <= target {
            delta_best = (mid, eo);
            lo = mid as i32 + 1;
        } else {
            hi = mid as i32 - 1;
        }
    }
    Ok(search_out(ladder[delta_best.0], delta_best.1, target))
}
