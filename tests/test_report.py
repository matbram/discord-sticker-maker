"""Tests for the honesty report, result shape, and CLI argument parsing."""
from __future__ import annotations

from encoder.cli.main import _human, build_parser
from encoder.core.result import EncodeReport, EncodeResult, LossLocus


def _base_report(**kw) -> EncodeReport:
    data = dict(
        input_path="x.mp4", input_kind="video", mode="cap", target_bytes=512 * 1024,
        achieved_bytes=500 * 1024, under_target=True, perceptually_lossless=True,
        perceptual_distance=0.001, metric_name="msssim+temporal", invisible_threshold=0.005,
        output_fps=24.0, n_frames=30, duration_ms=1250, engine_used="ffmpeg-palette",
        lever_setting={"colors": 128},
    )
    data.update(kw)
    return EncodeReport(**data)


def test_report_round_trips_and_has_honesty_fields():
    r = _base_report()
    d = r.model_dump()
    assert EncodeReport.model_validate(d) == r
    for field in ("perceptually_lossless", "under_target", "achieved_bytes", "output_fps",
                  "loss_locus", "stop_reason", "attempts", "fovea_version", "warnings"):
        assert field in d


def test_lossless_has_no_loss_locus():
    assert _base_report(perceptually_lossless=True, loss_locus=None).loss_locus is None


def test_visible_trade_off_carries_loss_locus():
    r = _base_report(
        perceptually_lossless=False, under_target=True,
        loss_locus=LossLocus(worst_frame=7, worst_frame_distance=0.04, region_hint="fast motion"),
    )
    d = r.model_dump()
    assert d["loss_locus"]["worst_frame"] == 7
    assert EncodeReport.model_validate(d).loss_locus.worst_frame == 7


def test_fovea_version_is_populated():
    assert _base_report().fovea_version


def test_encode_result_shape():
    res = EncodeResult(path="o.gif", size_bytes=1000, perceptually_lossless=False,
                       output_fps=15.0, notes=["softening near frame 3"])
    assert res.size_bytes == 1000
    assert res.notes == ["softening near frame 3"]


def test_cli_parses_encode_args():
    args = build_parser().parse_args(
        ["encode", "in.mp4", "--target-size", "8MB", "--mode", "invisible", "-o", "out.gif"]
    )
    assert args.input == "in.mp4"
    assert args.target_size == "8MB"
    assert args.mode == "invisible"
    assert args.output == "out.gif"


def test_human_size_formatting():
    assert _human(512 * 1024).endswith("KB")
    assert _human(3 * 1024 * 1024).endswith("MB")
    assert _human(500) == "500B"
