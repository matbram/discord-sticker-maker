//! Fovea native encoder.
//!
//! Pure-Rust GIF encoder that breaks the single-global-palette ceiling: per-frame
//! local color tables (perceptual quantization via `imagequant`) plus perceptual
//! OKLab inter-frame delta so unchanged pixels cost ~0 bytes. Exposed to Python as
//! the `fovea_native` extension module (built with maturin, `python` feature).

pub mod apng;
pub mod encode;
pub mod oklab;

pub use apng::{encode_apng, ApngOpts, ApngOut};
pub use encode::{encode_frames, encode_search, EncodeOpts, EncodeOut, SearchOpts, SearchOut};

#[cfg(feature = "python")]
mod python;
