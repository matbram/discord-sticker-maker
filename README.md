# Discord Media Studio + Fovea encoder

Upload an image / GIF / video → get Discord-ready **stickers, emoji, and GIFs**,
each optimized to fit the platform's size limits.

The GIF/emoji outputs are produced by **Fovea**, a from-scratch
perceptually-lossless GIF encoder (the project's IP) that hits a hard byte
target while balancing frames and colors. The sticker output is APNG (legacy
encoder). Backend: FastAPI (`backend/`). Frontend: Svelte (`frontend/`). Encoder:
`encoder/` (+ `bench/`). Deployed on Railway.

## 👉 Resuming work / onboarding

**Read [`docs/STATE.md`](docs/STATE.md) first** — it's the living handoff doc:
current state, how the encoder works (vivid internals), known gotchas, the
roadmap (M2–M5), and the dev/deploy runbook. Other docs:
[`fovea-spec.md`](docs/fovea-spec.md) ·
[`architecture.md`](docs/architecture.md) ·
[`metrics.md`](docs/metrics.md) ·
[`bench.md`](docs/bench.md) · [`cli.md`](docs/cli.md).

## Quickstart

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q                                   # encoder unit tests (no binaries needed)
fovea encode IN.mp4 --target-size 512KB -o out.gif
```

External tools the encoder shells out to (in the Docker image; install locally
for the gated integration test): `ffmpeg`, `gifsicle` (gifski optional).
