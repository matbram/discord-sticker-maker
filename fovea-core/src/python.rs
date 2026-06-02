//! PyO3 bindings: the `fovea_native` Python extension module.
//!
//! Frames are passed as a list of contiguous RGBA byte buffers (one per frame)
//! plus width/height/delays, avoiding any numpy build coupling. The encode runs
//! with the GIL released (`allow_threads`) so the synchronous server stays responsive.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::encode::{encode_frames, encode_search, EncodeOpts, SearchOpts};

#[pyfunction]
#[pyo3(signature = (
    frames, width, height, delays_cs,
    max_colors = 256, dithering = 1.0, quality_min = 0, quality_max = 100,
    speed = 4, delta_threshold = 0.0, loop_count = 0,
))]
#[allow(clippy::too_many_arguments)]
fn encode<'py>(
    py: Python<'py>,
    frames: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    delays_cs: Vec<u16>,
    max_colors: u16,
    dithering: f32,
    quality_min: u8,
    quality_max: u8,
    speed: i32,
    delta_threshold: f32,
    loop_count: u16,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = EncodeOpts {
        width,
        height,
        max_colors,
        dithering,
        quality_min,
        quality_max,
        speed,
        delta_threshold,
        loop_count,
    };
    let out = py
        .allow_threads(|| encode_frames(&frames, &delays_cs, &opts))
        .map_err(PyValueError::new_err)?;

    let dict = PyDict::new_bound(py);
    dict.set_item("gif", PyBytes::new_bound(py, &out.gif))?;
    dict.set_item("colors_per_frame", out.colors_per_frame)?;
    dict.set_item("distinct_colors", out.distinct_colors)?;
    dict.set_item("reused_pixels", out.reused_pixels)?;
    dict.set_item("total_pixels", out.total_pixels)?;
    dict.set_item("mode", out.mode)?;
    Ok(dict)
}

/// Byte-target search done entirely in Rust: bisect the per-frame color budget to
/// the largest palette that fits `target_bytes`, then encode the winner. Returns the
/// fitting GIF (or the smallest possible) plus stats. This is the fast path the
/// encoder uses so the search always completes and never ships an over-budget file.
#[pyfunction]
#[pyo3(signature = (
    frames, width, height, delays_cs, target_bytes,
    max_colors = 256, min_colors = 2, dithering = 1.0, quality_min = 0, quality_max = 100,
    speed = 5, delta_threshold = 0.0, loop_count = 0,
))]
#[allow(clippy::too_many_arguments)]
fn search<'py>(
    py: Python<'py>,
    frames: Vec<Vec<u8>>,
    width: u32,
    height: u32,
    delays_cs: Vec<u16>,
    target_bytes: u64,
    max_colors: u16,
    min_colors: u16,
    dithering: f32,
    quality_min: u8,
    quality_max: u8,
    speed: i32,
    delta_threshold: f32,
    loop_count: u16,
) -> PyResult<Bound<'py, PyDict>> {
    let sopts = SearchOpts {
        width,
        height,
        target_bytes,
        max_colors,
        min_colors,
        dithering,
        quality_min,
        quality_max,
        speed,
        delta_threshold,
        loop_count,
    };
    let out = py
        .allow_threads(|| encode_search(&frames, &delays_cs, &sopts))
        .map_err(PyValueError::new_err)?;

    let dict = PyDict::new_bound(py);
    dict.set_item("gif", PyBytes::new_bound(py, &out.gif))?;
    dict.set_item("colors", out.colors)?;
    dict.set_item("colors_per_frame", out.colors_per_frame)?;
    dict.set_item("distinct_colors", out.distinct_colors)?;
    dict.set_item("reused_pixels", out.reused_pixels)?;
    dict.set_item("total_pixels", out.total_pixels)?;
    dict.set_item("mode", out.mode)?;
    dict.set_item("under_budget", out.under_budget)?;
    Ok(dict)
}

#[pymodule]
fn fovea_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(encode, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
