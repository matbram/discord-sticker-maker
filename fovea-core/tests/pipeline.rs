//! End-to-end tests for the native encoder, asserting the properties the product
//! cares about: every frame kept, the global-256-color ceiling broken, and static
//! content reused (near-zero cost) on opaque clips.

use fovea_native::{encode_frames, encode_search, EncodeOpts, SearchOpts};

const W: u32 = 64;
const H: u32 = 64;

/// A smooth opaque gradient whose hue depends on the frame index, so each frame
/// genuinely needs its *own* rich palette and the cross-frame union exceeds 256.
fn gradient_frame(idx: usize) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let mut buf = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 4;
            buf[p] = ((x * 255 / w) as usize + idx * 17) as u8;
            buf[p + 1] = ((y * 255 / h) as usize + idx * 31) as u8;
            buf[p + 2] = ((x + y) * 255 / (w + h)) as u8 ^ (idx as u8).wrapping_mul(53);
            buf[p + 3] = 255;
        }
    }
    buf
}

/// Static smooth-gradient background (same every frame, like a sky behind a
/// subject) with one small bright square that moves each frame. The gradient is
/// expensive to re-encode every frame in full mode but is sent once under delta.
fn moving_square_frame(idx: usize) -> Vec<u8> {
    let (w, h) = (W as usize, H as usize);
    let mut buf = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let p = (y * w + x) * 4;
            buf[p] = (x * 255 / w) as u8;
            buf[p + 1] = (y * 255 / h) as u8;
            buf[p + 2] = 128;
            buf[p + 3] = 255;
        }
    }
    let sx = (idx * 3) % (w - 8);
    let sy = (idx * 2) % (h - 8);
    for yy in 0..8 {
        for xx in 0..8 {
            let p = ((sy + yy) * w + (sx + xx)) * 4;
            buf[p] = 240;
            buf[p + 1] = 80;
            buf[p + 2] = 40;
            buf[p + 3] = 255;
        }
    }
    buf
}

/// Decode a GIF and return how many image frames it contains.
fn count_gif_frames(bytes: &[u8]) -> usize {
    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::RGBA);
    let mut decoder = opts.read_info(std::io::Cursor::new(bytes)).unwrap();
    let mut n = 0;
    while decoder.read_next_frame().unwrap().is_some() {
        n += 1;
    }
    n
}

#[test]
fn full_mode_keeps_all_frames_and_breaks_the_256_color_ceiling() {
    let frames: Vec<Vec<u8>> = (0..8).map(gradient_frame).collect();
    let delays = vec![10u16; frames.len()];
    let opts = EncodeOpts {
        width: W,
        height: H,
        max_colors: 256,
        delta_threshold: 0.0, // force full (independent per-frame palettes)
        ..Default::default()
    };
    let out = encode_frames(&frames, &delays, &opts).unwrap();

    assert_eq!(out.mode, "full");
    assert_eq!(out.colors_per_frame.len(), frames.len());
    assert_eq!(count_gif_frames(&out.gif), frames.len(), "all frames kept");
    // The whole point: more than 256 distinct colors across the file, which a single
    // global palette physically cannot represent.
    assert!(
        out.distinct_colors > 256,
        "distinct colors across file = {} (should exceed the 256 global ceiling)",
        out.distinct_colors
    );
    assert_eq!(&out.gif[0..6], b"GIF89a");
}

#[test]
fn delta_mode_reuses_static_pixels() {
    let frames: Vec<Vec<u8>> = (0..12).map(moving_square_frame).collect();
    let delays = vec![5u16; frames.len()];
    let opts = EncodeOpts {
        width: W,
        height: H,
        max_colors: 256,
        delta_threshold: 0.02, // perceptual JND-ish
        ..Default::default()
    };
    let out = encode_frames(&frames, &delays, &opts).unwrap();

    assert_eq!(out.mode, "delta");
    assert_eq!(count_gif_frames(&out.gif), frames.len(), "all frames kept");
    // The background never changes, so the vast majority of pixels are reused.
    let reuse = out.reused_pixels as f64 / out.total_pixels as f64;
    assert!(reuse > 0.7, "reuse fraction = {reuse:.3} (expected mostly static)");
}

#[test]
fn delta_keeps_the_file_far_smaller_than_full_for_static_content() {
    let frames: Vec<Vec<u8>> = (0..16).map(moving_square_frame).collect();
    let delays = vec![5u16; frames.len()];
    let base = EncodeOpts {
        width: W,
        height: H,
        max_colors: 256,
        ..Default::default()
    };
    let full = encode_frames(
        &frames,
        &delays,
        &EncodeOpts { delta_threshold: 0.0, ..base.clone() },
    )
    .unwrap();
    let delta = encode_frames(
        &frames,
        &delays,
        &EncodeOpts { delta_threshold: 0.02, ..base },
    )
    .unwrap();
    // Delta sends the gradient once; full re-sends it every frame. Expect a clear win.
    assert!(
        delta.gif.len() * 4 < full.gif.len() * 3,
        "delta {} should be clearly smaller than full {}",
        delta.gif.len(),
        full.gif.len()
    );
}

#[test]
fn transparency_forces_full_mode_and_is_correct() {
    let (w, h) = (W as usize, H as usize);
    // A frame with a transparent hole forces full mode even with a delta threshold.
    let mut frames: Vec<Vec<u8>> = (0..4).map(moving_square_frame).collect();
    for f in frames.iter_mut() {
        for p in 0..(w * h / 4) {
            f[p * 4 + 3] = 0; // top quarter transparent
        }
    }
    let delays = vec![10u16; frames.len()];
    let opts = EncodeOpts {
        width: W,
        height: H,
        delta_threshold: 0.05,
        ..Default::default()
    };
    let out = encode_frames(&frames, &delays, &opts).unwrap();
    assert_eq!(out.mode, "full", "alpha matte must fall back to full frames");
    assert_eq!(count_gif_frames(&out.gif), frames.len());
}

fn search_opts(target: u64, delta: f32) -> SearchOpts {
    SearchOpts {
        width: W,
        height: H,
        target_bytes: target,
        max_colors: 256,
        min_colors: 2,
        dithering: 1.0,
        quality_min: 0,
        quality_max: 100,
        speed: 6,
        delta_threshold: delta,
        loop_count: 0,
    }
}

#[test]
fn search_always_fits_the_target_and_keeps_all_frames() {
    // 16 rich full-motion frames: at 256 colors this busts a tight target, so the
    // search must drop the per-frame color budget until it fits — but keep every frame.
    let frames: Vec<Vec<u8>> = (0..16).map(gradient_frame).collect();
    let delays = vec![10u16; frames.len()];

    let big = encode_frames(
        &frames,
        &delays,
        &EncodeOpts { width: W, height: H, max_colors: 256, ..Default::default() },
    )
    .unwrap();
    let target = (big.gif.len() as u64) / 2; // force a real reduction

    let out = encode_search(&frames, &delays, &search_opts(target, 0.0)).unwrap();
    assert!(out.under_budget, "search must return a fitting result");
    assert!(out.gif.len() as u64 <= target, "{} > {}", out.gif.len(), target);
    assert_eq!(count_gif_frames(&out.gif), frames.len(), "all frames kept");
    assert!(out.colors >= 2 && out.colors <= 256);
}

#[test]
fn search_returns_smallest_when_nothing_fits() {
    // An impossibly tiny target: the search must still return *something* (the minimal
    // encode), flagged as not under budget, rather than spinning or failing.
    let frames: Vec<Vec<u8>> = (0..8).map(gradient_frame).collect();
    let delays = vec![10u16; frames.len()];
    let out = encode_search(&frames, &delays, &search_opts(50, 0.0)).unwrap();
    assert!(!out.under_budget);
    assert_eq!(out.colors, 2, "fell back to the minimum color budget");
    assert_eq!(count_gif_frames(&out.gif), frames.len());
}

#[test]
fn lower_color_budget_yields_smaller_file() {
    let frames: Vec<Vec<u8>> = (0..6).map(gradient_frame).collect();
    let delays = vec![10u16; frames.len()];
    let base = EncodeOpts {
        width: W,
        height: H,
        delta_threshold: 0.0,
        ..Default::default()
    };
    let many = encode_frames(&frames, &delays, &EncodeOpts { max_colors: 256, ..base.clone() }).unwrap();
    let few = encode_frames(&frames, &delays, &EncodeOpts { max_colors: 16, ..base }).unwrap();
    assert!(
        few.gif.len() < many.gif.len(),
        "16-color {} should be smaller than 256-color {}",
        few.gif.len(),
        many.gif.len()
    );
    // Even at 16 colors *per frame*, the cross-frame union beats a 16-color global palette.
    assert!(few.distinct_colors > 16);
}
