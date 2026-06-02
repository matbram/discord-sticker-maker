# ---------- build the frontend ----------
FROM node:22-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---------- build the native encoder wheel (Rust + maturin) ----------
# fovea_native is the per-frame-local-palette + perceptual-delta GIF engine. It is
# built as an abi3 wheel here and pip-installed into the runtime below.
FROM rust:1-slim-bookworm AS rustbuild
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir --break-system-packages "maturin>=1,<2"
WORKDIR /build
COPY fovea-core ./fovea-core
RUN cd fovea-core && maturin build --release -i python3 --out /wheels

# ---------- backend runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/models \
    OMP_NUM_THREADS=2 \
    STATIC_DIR=/app/frontend/dist

# ffmpeg = decode any video/animated; pngquant = APNG color optimization;
# apngasm = inter-frame APNG compression (more frames under 512KB); libheif = HEIC;
# gifsicle = a Fovea encoder engine (lossy-LZW GIF post-pass). gifski is optional
# (no apt package); add it via a release binary or `cargo install gifski` if the
# opaque video->GIF path is needed — Fovea degrades gracefully without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg pngquant apngasm libheif1 gifsicle \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

# Native Fovea encoder (abi3 wheel from the rustbuild stage). Optional at runtime:
# FoveaNativeEngine falls back to the ffmpeg engines if this import ever fails, so a
# build hiccup here can never break the app.
COPY --from=rustbuild /wheels /tmp/wheels
# --force-reinstall: never let a cached same-version wheel shadow a fresh build. Then verify
# the freshly-built extension actually exposes the truecolor-APNG entrypoint the sticker path
# depends on — fail the build loudly rather than silently shipping a stale wheel (which would
# make every animated sticker fall back to the washed-out legacy palette).
RUN pip install --force-reinstall --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels \
    && python -c "import fovea_native as f; assert hasattr(f, 'encode_apng'), 'stale fovea_native wheel: encode_apng missing'; print('fovea_native', f.__version__, 'encode_apng OK')"

COPY backend/app ./app
# Fovea encoder package (the backend GIF path imports `encoder`). Its deps
# (numpy/Pillow/pydantic) are already satisfied by backend/requirements.txt;
# /app is on sys.path so `import encoder` resolves without a separate install.
COPY encoder ./encoder
COPY --from=frontend /fe/dist ./frontend/dist

# Bake the default background-removal model into the image so the first
# request isn't slowed by a download. Non-fatal if it can't fetch at build.
RUN python -c "from app.pipeline import bg_removal; bg_removal.warmup([bg_removal.DEFAULT_MODEL])" || true

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
