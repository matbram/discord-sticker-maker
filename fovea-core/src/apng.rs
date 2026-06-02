//! Truecolor APNG encoder with perceptual OKLab inter-frame delta.
//!
//! APNG is PNG-based: 24-bit truecolor + 8-bit alpha + DEFLATE. Its color type lives in
//! one global IHDR, so the GIF "per-frame local palette" trick can't port — but truecolor
//! means there's **no palette and therefore no washout/sepia** at all. The size lever is
//! instead the inter-frame delta: keep an OKLab canvas and redraw only the pixels the eye
//! can see change (ΔE >= ~1 JND), as an APNG sub-frame over the changed bounding box.
//!
//! We use `blend_op = SOURCE` over the bounding box (not GIF's transparent-index + OVER):
//! OVER can only paint on top, so it can't erase a pixel back to transparent when a cut-out
//! subject moves away — SOURCE replaces the whole sub-rect, which is correct for every
//! transition (color change, reveal, erase) and still small because the sub-rect is the
//! tight bbox of perceptible change.
//!
//! Phase 2 will add perceptual entropy-reduction transforms (denoise / chroma / adaptive
//! precision) so dense full-frame stickers fit the byte budget at full color. This module
//! is the foundation: full-color + every-frame + delta reuse.

use png::{BitDepth, BlendOp, ColorType, Compression, DisposeOp, Encoder};
use rayon::prelude::*;

use crate::oklab::{delta_e_sq, rgb_to_oklab, srgb_to_linear_lut, Lab};

// ---------------------------------------------------------------------------------------
// Phase 2 — perceptual entropy reduction (the "fool perception" part).
//
// APNG is lossless DEFLATE: a full-color clip can't fit a small budget *losslessly*. We
// manufacture the lossiness perceptually — spend the budget on entropy the eye won't miss:
//   * edge-aware denoise: grain is incompressible high-frequency noise; smoothing it (while
//     PRESERVING edges) turns random PNG-filter residuals into tiny, repetitive ones that
//     DEFLATE crushes. The eye doesn't track individual grain, so this is ~invisible.
//   * chroma reduction: the eye tolerates chroma blur far more than luma, so coarsening the
//     chroma channels (YCoCg) drops entropy with no perceptible change.
// A single `strength` knob (driven by the metric-guided search in the bridge) scales both.
// ---------------------------------------------------------------------------------------

/// Alpha-weighted separable box blur of the RGB channels (alpha copied through). Weighting by
/// opacity keeps the transparent background from bleeding dark pixels into the subject edge.
fn box_blur_rgb_alpha(rgba: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    // Horizontal pass -> weighted sums; vertical pass -> final.
    let n = w * h;
    let mut hr = vec![0f32; n]; // sum R*wt
    let mut hg = vec![0f32; n];
    let mut hb = vec![0f32; n];
    let mut hw_ = vec![0f32; n]; // sum wt
    for y in 0..h {
        for x in 0..w {
            let (mut sr, mut sg, mut sb, mut sw) = (0f32, 0f32, 0f32, 0f32);
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            for xx in x0..=x1 {
                let i = (y * w + xx) * 4;
                let wt = if rgba[i + 3] >= 128 { 1.0 } else { 0.0 };
                sr += rgba[i] as f32 * wt;
                sg += rgba[i + 1] as f32 * wt;
                sb += rgba[i + 2] as f32 * wt;
                sw += wt;
            }
            let p = y * w + x;
            hr[p] = sr;
            hg[p] = sg;
            hb[p] = sb;
            hw_[p] = sw;
        }
    }
    let mut out = rgba.to_vec();
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            if rgba[p * 4 + 3] < 128 {
                continue;
            }
            let (mut sr, mut sg, mut sb, mut sw) = (0f32, 0f32, 0f32, 0f32);
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            for yy in y0..=y1 {
                let q = yy * w + x;
                sr += hr[q];
                sg += hg[q];
                sb += hb[q];
                sw += hw_[q];
            }
            if sw > 0.0 {
                let i = p * 4;
                out[i] = (sr / sw).round().clamp(0.0, 255.0) as u8;
                out[i + 1] = (sg / sw).round().clamp(0.0, 255.0) as u8;
                out[i + 2] = (sb / sw).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

/// Edge-aware denoise gated by perception: blend each opaque pixel toward its (alpha-weighted)
/// blur by how *imperceptible* that blur is. The OKLab ΔE between a pixel and its blur is tiny
/// for grain and smooth gradients (their blur looks the same → smooth them away, which DEFLATE
/// loves) but large at real edges (the blur looks different → preserve them). `strength` 0..1
/// scales the blend. This is the key entropy-reducer: grain is incompressible noise the eye
/// doesn't track, so removing it shrinks the deflated truecolor dramatically and invisibly.
fn denoise_frame(rgba: &[u8], w: usize, h: usize, strength: f32, lut: &[f32; 256]) -> Vec<u8> {
    if strength <= 0.0 {
        return rgba.to_vec();
    }
    // ΔE below `flat` = grain/gradient (smooth fully); above `edge` = real edge (keep);
    // linear ramp between. In OKLab units (~0.02 ≈ 1 JND).
    const FLAT: f32 = 0.045;
    const EDGE: f32 = 0.11;
    // Radius and edge-preservation scale with strength: low strength = small radius,
    // edge-PRESERVING grain removal (imperceptible); high strength = large radius and the
    // edge gate is progressively overridden, so at strength→1 the frame is heavily blurred
    // (visible, but full-color + full-frame) — this is the guaranteed-fit last resort.
    // Radius grows with strength; strength may exceed 1.0 as a guaranteed-fit last resort
    // (heavier blur -> tiny file) for pathological full-frame-motion content.
    let r = (1.0 + strength * 4.0).round().max(1.0) as usize;
    let edge_override = (strength * strength).min(1.0); // 0 (preserve edges) .. 1 (smooth all)
    let blur = box_blur_rgb_alpha(rgba, w, h, r);
    let mut out = rgba.to_vec();
    let lerp = |a: u8, b: u8, t: f32| -> u8 {
        (a as f32 + (b as f32 - a as f32) * t).round().clamp(0.0, 255.0) as u8
    };
    for p in 0..w * h {
        let i = p * 4;
        if rgba[i + 3] < 128 {
            continue;
        }
        let o = rgb_to_oklab(rgba[i], rgba[i + 1], rgba[i + 2], lut);
        let b = rgb_to_oklab(blur[i], blur[i + 1], blur[i + 2], lut);
        let de = delta_e_sq(&o, &b).sqrt();
        let t = if de <= FLAT {
            1.0
        } else if de >= EDGE {
            0.0
        } else {
            1.0 - (de - FLAT) / (EDGE - FLAT)
        };
        // Blend from edge-preserving (t) toward smooth-everything (1.0) as strength rises.
        let eff_t = t + (1.0 - t) * edge_override;
        let wgt = (strength * eff_t).min(1.0);
        out[i] = lerp(rgba[i], blur[i], wgt);
        out[i + 1] = lerp(rgba[i + 1], blur[i + 1], wgt);
        out[i + 2] = lerp(rgba[i + 2], blur[i + 2], wgt);
    }
    out
}

/// Coarsen chroma via the reversible YCoCg-R transform: quantize Co/Cg to a step, keep luma.
/// The eye tolerates chroma quantization far more than luma, so this drops entropy invisibly.
fn chroma_reduce(rgba: &[u8], step: u8) -> Vec<u8> {
    if step <= 1 {
        return rgba.to_vec();
    }
    let s = step as i32;
    let q = |v: i32| -> i32 { ((v as f32 / s as f32).round() as i32) * s };
    let mut out = rgba.to_vec();
    for px in out.chunks_exact_mut(4) {
        if px[3] < 128 {
            continue;
        }
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        // RGB -> YCoCg-R
        let co = r - b;
        let t = b + (co >> 1);
        let cg = g - t;
        let y = t + (cg >> 1);
        // quantize chroma only
        let co = q(co);
        let cg = q(cg);
        // YCoCg-R -> RGB
        let t = y - (cg >> 1);
        let g2 = cg + t;
        let b2 = t - (co >> 1);
        let r2 = b2 + co;
        px[0] = r2.clamp(0, 255) as u8;
        px[1] = g2.clamp(0, 255) as u8;
        px[2] = b2.clamp(0, 255) as u8;
    }
    out
}

/// Apply the perceptual transforms to every frame (parallel). `denoise` 0..1, `chroma_step`
/// 1=off. Returns new frames; the delta encoder then runs on these.
fn preprocess(frames: &[Vec<u8>], w: usize, h: usize, denoise: f32, chroma_step: u8) -> Vec<Vec<u8>> {
    if denoise <= 0.0 && chroma_step <= 1 {
        return frames.to_vec();
    }
    let lut = srgb_to_linear_lut();
    frames
        .par_iter()
        .map(|f| {
            let d = denoise_frame(f, w, h, denoise, &lut);
            chroma_reduce(&d, chroma_step)
        })
        .collect()
}

/// Options for the truecolor APNG encode.
pub struct ApngOpts {
    pub width: u32,
    pub height: u32,
    /// OKLab ΔE threshold: pixels closer than this to the displayed canvas are reused
    /// (not redrawn). ~0.02 is roughly one just-noticeable difference.
    pub delta_threshold: f32,
    /// Alpha change (0..255) large enough to force a redraw even if RGB is unchanged.
    pub alpha_threshold: u8,
    /// 0 = loop forever.
    pub loop_count: u16,
    /// 0=none,1=fastest,2=fast,3=balanced,4=high (DEFLATE effort).
    pub compression: u8,
    /// Perceptual entropy reduction (Phase 2): edge-aware denoise strength 0..1 (grain removal).
    pub denoise: f32,
    /// Chroma quantization step (1 = off; higher = coarser chroma, lower entropy).
    pub chroma_step: u8,
}

/// Result of an APNG encode.
pub struct ApngOut {
    pub png: Vec<u8>,
    pub reused_pixels: u64,
    pub total_pixels: u64,
    /// Frames that actually redrew something (vs reused the whole canvas).
    pub changed_frames: u32,
}

fn map_compression(level: u8) -> Compression {
    match level {
        0 => Compression::NoCompression,
        1 => Compression::Fastest,
        2 => Compression::Fast,
        3 => Compression::Balanced,
        _ => Compression::High,
    }
}

fn validate(frames: &[Vec<u8>], delays: &[u16], w: usize, h: usize) -> Result<(), String> {
    if frames.is_empty() {
        return Err("no frames".into());
    }
    if w == 0 || h == 0 || w > u32::MAX as usize || h > u32::MAX as usize {
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

/// Encode RGBA frames to a truecolor APNG, reusing perceptually-unchanged pixels.
///
/// `delays_cs` are centiseconds (APNG delay = delay/100 s), matching the GIF binding.
pub fn encode_apng(frames: &[Vec<u8>], delays_cs: &[u16], opts: &ApngOpts) -> Result<ApngOut, String> {
    let w = opts.width as usize;
    let h = opts.height as usize;
    let npix = w * h;
    validate(frames, delays_cs, w, h)?;

    // Phase 2: perceptually reduce entropy (denoise grain, coarsen chroma) so truecolor fits
    // the byte budget. No-op when denoise=0 and chroma_step<=1 (Phase 1 lossless behavior).
    let processed = preprocess(frames, w, h, opts.denoise, opts.chroma_step);
    let frames: &[Vec<u8>] = &processed;

    let lut = srgb_to_linear_lut();
    let thr2 = opts.delta_threshold * opts.delta_threshold;
    let alpha_thr = opts.alpha_threshold as i16;
    let n = frames.len();

    let mut out: Vec<u8> = Vec::new();
    let mut reused: u64 = 0;
    let mut changed_frames: u32 = 0;
    {
        let mut enc = Encoder::new(&mut out, opts.width, opts.height);
        enc.set_color(ColorType::Rgba);
        enc.set_depth(BitDepth::Eight);
        enc.set_compression(map_compression(opts.compression));
        enc.set_animated(n as u32, opts.loop_count as u32)
            .map_err(|e| format!("apng set_animated: {e}"))?;
        let mut writer = enc.write_header().map_err(|e| format!("apng header: {e}"))?;

        // Displayed canvas (what a player currently shows) in RGBA + OKLab.
        let mut canvas = frames[0].clone();
        let mut canvas_lab = vec![Lab::default(); npix];
        for p in 0..npix {
            let i = p * 4;
            canvas_lab[p] = rgb_to_oklab(canvas[i], canvas[i + 1], canvas[i + 2], &lut);
        }

        // Frame 0: full image, SOURCE.
        writer.set_dispose_op(DisposeOp::None).map_err(|e| format!("apng f0 dispose: {e}"))?;
        writer.set_blend_op(BlendOp::Source).map_err(|e| format!("apng f0 blend: {e}"))?;
        writer.set_frame_delay(delays_cs[0].max(1), 100).map_err(|e| format!("apng f0 delay: {e}"))?;
        writer.write_image_data(&frames[0]).map_err(|e| format!("apng write f0: {e}"))?;

        for i in 1..n {
            let src = &frames[i];
            let (mut min_x, mut min_y, mut max_x, mut max_y) = (w, h, 0usize, 0usize);
            let mut changed = 0usize;
            for y in 0..h {
                for x in 0..w {
                    let p = y * w + x;
                    let idx = p * 4;
                    let a = src[idx + 3] as i16;
                    let ca = canvas[idx + 3] as i16;
                    // Two transparent pixels look identical regardless of RGB.
                    if a < 128 && ca < 128 {
                        continue;
                    }
                    let alpha_changed = (a - ca).abs() > alpha_thr;
                    let lab = rgb_to_oklab(src[idx], src[idx + 1], src[idx + 2], &lut);
                    if alpha_changed || delta_e_sq(&lab, &canvas_lab[p]) >= thr2 {
                        changed += 1;
                        min_x = min_x.min(x);
                        min_y = min_y.min(y);
                        max_x = max_x.max(x);
                        max_y = max_y.max(y);
                    }
                }
            }
            reused += (npix - changed) as u64;
            let delay = delays_cs[i].max(1);

            if changed == 0 {
                // Nothing the eye can see changed: a 1×1 fully-transparent OVER frame is a
                // no-op composite (keeps the canvas) and costs almost nothing.
                // Reset position first so the new (smaller) dimension can't clash with the
                // previous frame's offset (the writer validates position+dimension <= image).
                writer.reset_frame_position().map_err(|e| format!("apng nop reset: {e}"))?;
                writer.set_frame_dimension(1, 1).map_err(|e| format!("apng nop dim: {e}"))?;
                writer.set_frame_position(0, 0).map_err(|e| format!("apng nop pos: {e}"))?;
                writer.set_dispose_op(DisposeOp::None).map_err(|e| format!("apng nop dispose: {e}"))?;
                writer.set_blend_op(BlendOp::Over).map_err(|e| format!("apng nop blend: {e}"))?;
                writer.set_frame_delay(delay, 100).map_err(|e| format!("apng nop delay: {e}"))?;
                writer.write_image_data(&[0, 0, 0, 0]).map_err(|e| format!("apng write nop {i}: {e}"))?;
                continue;
            }
            changed_frames += 1;

            let bw = max_x - min_x + 1;
            let bh = max_y - min_y + 1;
            let mut sub = vec![0u8; bw * bh * 4];
            for yy in 0..bh {
                for xx in 0..bw {
                    let gp = (min_y + yy) * w + (min_x + xx);
                    let gi = gp * 4;
                    let sp = (yy * bw + xx) * 4;
                    sub[sp] = src[gi];
                    sub[sp + 1] = src[gi + 1];
                    sub[sp + 2] = src[gi + 2];
                    sub[sp + 3] = src[gi + 3];
                    // The whole bbox is redrawn (SOURCE), so the canvas there becomes frame i.
                    canvas[gi] = src[gi];
                    canvas[gi + 1] = src[gi + 1];
                    canvas[gi + 2] = src[gi + 2];
                    canvas[gi + 3] = src[gi + 3];
                    canvas_lab[gp] = rgb_to_oklab(src[gi], src[gi + 1], src[gi + 2], &lut);
                }
            }
            writer.reset_frame_position().map_err(|e| format!("apng reset {i}: {e}"))?;
            writer.set_frame_dimension(bw as u32, bh as u32).map_err(|e| format!("apng dim {i}: {e}"))?;
            writer.set_frame_position(min_x as u32, min_y as u32).map_err(|e| format!("apng pos {i}: {e}"))?;
            writer.set_dispose_op(DisposeOp::None).map_err(|e| format!("apng dispose {i}: {e}"))?;
            writer.set_blend_op(BlendOp::Source).map_err(|e| format!("apng blend {i}: {e}"))?;
            writer.set_frame_delay(delay, 100).map_err(|e| format!("apng delay {i}: {e}"))?;
            writer.write_image_data(&sub).map_err(|e| format!("apng write {i}: {e}"))?;
        }

        writer.finish().map_err(|e| format!("apng finish: {e}"))?;
    }

    Ok(ApngOut {
        png: out,
        reused_pixels: reused,
        total_pixels: npix as u64 * n as u64,
        changed_frames,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, rgba: [u8; 4]) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for p in 0..w * h {
            v[p * 4..p * 4 + 4].copy_from_slice(&rgba);
        }
        v
    }

    #[test]
    fn encodes_apng_signature_and_actl() {
        let w = 8;
        let h = 8;
        let f0 = solid(w, h, [200, 100, 50, 255]);
        let mut f1 = f0.clone();
        // change a 2x2 block
        for y in 2..4 {
            for x in 2..4 {
                let i = (y * w + x) * 4;
                f1[i..i + 4].copy_from_slice(&[10, 20, 30, 255]);
            }
        }
        let opts = ApngOpts {
            width: w as u32,
            height: h as u32,
            delta_threshold: 0.02,
            alpha_threshold: 24,
            loop_count: 0,
            compression: 4,
            denoise: 0.0,
            chroma_step: 1,
        };
        let out = encode_apng(&[f0, f1], &[10, 10], &opts).unwrap();
        // PNG signature
        assert_eq!(&out.png[0..8], &[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n']);
        // contains acTL (animation control) and fcTL/fdAT
        let has = |needle: &[u8]| out.png.windows(4).any(|w| w == needle);
        assert!(has(b"acTL"), "missing acTL");
        assert!(has(b"fcTL"), "missing fcTL");
        assert!(has(b"fdAT"), "missing fdAT");
        // most pixels reused on frame 1 (only 4 changed out of 64)
        assert!(out.reused_pixels >= 60, "reused={}", out.reused_pixels);
        assert_eq!(out.changed_frames, 1);
    }

    #[test]
    fn unchanged_frames_are_reused() {
        let w = 6;
        let h = 6;
        let f = solid(w, h, [12, 34, 56, 255]);
        let opts = ApngOpts {
            width: w as u32,
            height: h as u32,
            delta_threshold: 0.02,
            alpha_threshold: 24,
            loop_count: 0,
            compression: 4,
            denoise: 0.0,
            chroma_step: 1,
        };
        let out = encode_apng(&[f.clone(), f.clone(), f], &[10, 10, 10], &opts).unwrap();
        assert_eq!(out.changed_frames, 0, "identical frames must not redraw");
    }

    #[test]
    fn decodes_back_with_png_reader() {
        let w = 16;
        let h = 16;
        let f0 = solid(w, h, [0, 128, 255, 255]);
        let mut f1 = f0.clone();
        for x in 0..w {
            let i = (5 * w + x) * 4;
            f1[i..i + 4].copy_from_slice(&[255, 0, 0, 255]);
        }
        let opts = ApngOpts {
            width: w as u32,
            height: h as u32,
            delta_threshold: 0.02,
            alpha_threshold: 24,
            loop_count: 0,
            compression: 4,
            denoise: 0.0,
            chroma_step: 1,
        };
        let out = encode_apng(&[f0, f1], &[10, 10], &opts).unwrap();
        let dec = png::Decoder::new(std::io::Cursor::new(&out.png));
        let reader = dec.read_info().unwrap();
        let info = reader.info();
        assert_eq!(info.color_type, ColorType::Rgba);
        assert_eq!(info.width, w as u32);
        assert!(info.animation_control.is_some(), "no acTL");
        assert_eq!(info.animation_control.unwrap().num_frames, 2);
    }
}
