"""Best-effort fetch of small, permissively-licensed REAL clips from GitHub.

This environment's network policy blocks general video CDNs (Blender, Wikimedia,
archive.org all return 403) but allows ``raw.githubusercontent.com``. So we pull a
tiny allowlist of GitHub-hosted clips and record their licenses. Every entry is
HEAD-probed first and only downloaded if it is reachable and within a size cap; any
failure logs and is skipped (the run always exits 0 so the synthetic path can take
over). Real footage trains the judge on real-content statistics; synthetic clips
(``synth_clips.py``) fill any category these can't cover.
"""
from __future__ import annotations

import json
import os
import subprocess

# (url, category, license). Prefer CC0/CC-BY; clips are never committed (gitignored).
ALLOWLIST = [
    ("https://raw.githubusercontent.com/mediaelement/mediaelement-files/master/big_buck_bunny.mp4",
     "video_clip", "CC-BY-3.0 (Blender Foundation, Big Buck Bunny)"),
    ("https://raw.githubusercontent.com/ietf-wg-cellar/matroska-test-files/master/test_files/test5.mkv",
     "video_clip", "Matroska test suite (freely distributable)"),
    ("https://raw.githubusercontent.com/PixarAnimationStudios/OpenSubdiv/release/documentation/images/osd_splash.png",
     "motion_graphics", "unused-placeholder"),  # probed; skipped unless it is a real clip
]

MAX_BYTES = 30 * 1024 * 1024


def _head(url: str) -> tuple[int, int]:
    """Return (http_status, content_length) via a HEAD probe; (0, 0) on failure."""
    try:
        out = subprocess.run(
            ["curl", "-sI", "--max-time", "20", url],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001
        return (0, 0)
    status, length = 0, 0
    for line in out.splitlines():
        ls = line.strip().lower()
        if line.startswith("HTTP/"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                status = int(parts[1])
        elif ls.startswith("content-length:"):
            try:
                length = int(ls.split(":", 1)[1].strip())
            except ValueError:
                pass
    return (status, length)


def fetch_clips(out_dir: str) -> list[dict]:
    """Download reachable allowlist clips into ``out_dir``; return license records."""
    os.makedirs(out_dir, exist_ok=True)
    records: list[dict] = []
    for url, category, lic in ALLOWLIST:
        ext = os.path.splitext(url)[1].lower()
        if ext not in (".mp4", ".gif", ".webm", ".mkv", ".mov"):
            print(f"fetch skip (not a clip): {url}")
            continue
        status, length = _head(url)
        if status != 200:
            print(f"fetch blocked/missing status={status or 'ERR'}: {url}")
            continue
        if length and length > MAX_BYTES:
            print(f"fetch skip (too big {length}B): {url}")
            continue
        name = f"real_{category}_{os.path.basename(url)}"
        dest = os.path.join(out_dir, name)
        try:
            rc = subprocess.run(
                ["curl", "-sL", "--max-time", "120", "-o", dest, url],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180,
            ).returncode
        except Exception:  # noqa: BLE001
            rc = 1
        if rc != 0 or not os.path.exists(dest) or os.path.getsize(dest) < 1024:
            print(f"fetch failed (download): {url}")
            if os.path.exists(dest):
                os.remove(dest)
            continue
        rec = {"file": name, "category": category, "license": lic, "source_url": url,
               "bytes": os.path.getsize(dest)}
        records.append(rec)
        print(f"fetch ok [{category}] {name} ({rec['bytes']} B) — {lic}")
    if records:
        with open(os.path.join(out_dir, "LICENSES.json"), "w") as fh:
            json.dump(records, fh, indent=2)
    else:
        print("fetch: no real clips reachable; synthetic clips will cover all categories.")
    return records


if __name__ == "__main__":
    import sys

    fetch_clips(sys.argv[1] if len(sys.argv) > 1 else "bench/corpus/train_clips")
