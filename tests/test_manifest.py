"""Tests for the corpus manifest, result records, and skip-clean bench behavior."""
from __future__ import annotations

import os

import pytest

from bench.manifest import ClipEntry, Manifest, clip_present, load_manifest, resolved_targets
from bench.records import ResultRecord, read_json, summary_table, write_csv, write_json

MANIFEST = os.path.join("bench", "corpus", "manifest.yaml")
CORPUS = os.path.join("bench", "corpus")


def test_committed_manifest_is_valid_and_covers_all_categories():
    m = load_manifest(MANIFEST)
    assert m.version == 1
    assert len(m.clips) >= 3
    cats = {c.category for c in m.clips}
    assert {"screen_recording", "video_clip", "motion_graphics"} <= cats


def test_all_clips_absent_on_fresh_checkout():
    m = load_manifest(MANIFEST)
    assert m.clips and all(not clip_present(c, CORPUS) for c in m.clips)


def test_resolved_targets_parse_to_bytes():
    m = load_manifest(MANIFEST)
    targets = resolved_targets(m.clips[0], m)
    assert all(isinstance(t, int) and t > 0 for t in targets)
    assert 256 * 1024 in targets


def test_per_clip_target_override():
    m = Manifest(clips=[ClipEntry(id="x", path="clips/x.mp4", category="video_clip",
                                  target_sizes=["1MB"])])
    assert resolved_targets(m.clips[0], m) == [1024 * 1024]


def test_invalid_manifest_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\nclips:\n  - id: a\n    path: p.mp4\n    category: nonsense\n")
    with pytest.raises(Exception):
        load_manifest(str(bad))
    missing_field = tmp_path / "missing.yaml"
    missing_field.write_text("clips:\n  - id: a\n    category: video_clip\n")  # no path
    with pytest.raises(Exception):
        load_manifest(str(missing_field))


def test_records_round_trip_and_summary(tmp_path):
    recs = [
        ResultRecord(clip_id="c", category="video_clip", engine="gifski", target_bytes=512 * 1024,
                     achieved_bytes=500 * 1024, under_target=True, lever_setting={"quality": 70},
                     distance=0.02, msssim=0.98, temporal=0.01, fps=24.0, n_frames=30,
                     metric_name="msssim+temporal", fovea_version="0.1.0"),
        ResultRecord(clip_id="c", category="video_clip", engine="ffmpeg-palette",
                     target_bytes=512 * 1024, skipped_reason="binary_missing:ffmpeg-palette"),
    ]
    jp, cp = str(tmp_path / "r.json"), str(tmp_path / "r.csv")
    write_json(recs, jp)
    write_csv(recs, cp)
    assert read_json(jp) == recs
    with open(cp) as fh:
        assert sum(1 for _ in fh) >= 3            # header + 2 rows
    text = summary_table(recs)
    assert "gifski" in text and "skipped" in text


def test_run_bench_skips_cleanly_with_no_clips(tmp_path):
    from bench.run import format_run_summary, run_bench

    records, meta = run_bench(MANIFEST, CORPUS, out_dir=str(tmp_path / "out"), max_attempts=4)
    assert records, "expected skip records, not an empty list"
    assert all(r.skipped_reason == "clip_missing" for r in records)
    assert "0 runnable" in format_run_summary(records, meta)
    assert (tmp_path / "out" / "results.json").exists()
    assert (tmp_path / "out" / "results.csv").exists()
