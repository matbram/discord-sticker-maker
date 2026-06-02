//! OKLab perceptual color space, used to decide *which pixels changed enough to
//! matter* between frames (the inter-frame delta threshold).
//!
//! OKLab (Björn Ottosson, 2020) is near-perceptually-uniform: a Euclidean
//! distance in (L, a, b) tracks how different two colors look to the human eye,
//! far better than raw RGB. We threshold that distance to reuse unchanged pixels,
//! so "below the threshold" genuinely means "the eye won't notice."

/// A color in OKLab.
#[derive(Clone, Copy, Debug, Default)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

/// 8-bit sRGB component -> linear light. Precomputed into a 256-entry LUT so the
/// per-pixel hot path is table lookups + a cube root, not `powf`.
pub fn srgb_to_linear_lut() -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    let mut i = 0;
    while i < 256 {
        let c = i as f32 / 255.0;
        lut[i] = if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        };
        i += 1;
    }
    lut
}

/// Convert an 8-bit sRGB triple to OKLab using a precomputed linearization LUT.
#[inline]
pub fn rgb_to_oklab(r: u8, g: u8, b: u8, lut: &[f32; 256]) -> Lab {
    let lr = lut[r as usize];
    let lg = lut[g as usize];
    let lb = lut[b as usize];

    let l = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb;
    let m = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb;
    let s = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb;

    let l_ = l.cbrt();
    let m_ = m.cbrt();
    let s_ = s.cbrt();

    Lab {
        l: 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        a: 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        b: 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    }
}

/// Squared OKLab distance. We compare against a squared threshold to avoid the
/// per-pixel `sqrt`.
#[inline]
pub fn delta_e_sq(x: &Lab, y: &Lab) -> f32 {
    let dl = x.l - y.l;
    let da = x.a - y.a;
    let db = x.b - y.b;
    dl * dl + da * da + db * db
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_colors_have_zero_distance() {
        let lut = srgb_to_linear_lut();
        let a = rgb_to_oklab(123, 45, 200, &lut);
        let b = rgb_to_oklab(123, 45, 200, &lut);
        assert!(delta_e_sq(&a, &b) < 1e-9);
    }

    #[test]
    fn black_white_is_about_one_in_lightness() {
        let lut = srgb_to_linear_lut();
        let black = rgb_to_oklab(0, 0, 0, &lut);
        let white = rgb_to_oklab(255, 255, 255, &lut);
        // OKLab L runs ~0..1 from black to white.
        assert!((white.l - 1.0).abs() < 0.02, "white L = {}", white.l);
        assert!(black.l.abs() < 0.02, "black L = {}", black.l);
    }

    #[test]
    fn near_colors_are_below_a_just_noticeable_threshold() {
        let lut = srgb_to_linear_lut();
        // One 8-bit step in a mid gray is imperceptible.
        let a = rgb_to_oklab(128, 128, 128, &lut);
        let b = rgb_to_oklab(129, 128, 128, &lut);
        assert!(delta_e_sq(&a, &b).sqrt() < 0.01);
        // A big jump is well above it.
        let c = rgb_to_oklab(200, 50, 50, &lut);
        assert!(delta_e_sq(&a, &c).sqrt() > 0.1);
    }
}
