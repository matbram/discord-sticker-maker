# ---------- build the frontend ----------
FROM node:22-slim AS frontend
WORKDIR /fe
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ---------- backend runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/models \
    OMP_NUM_THREADS=2 \
    STATIC_DIR=/app/frontend/dist

# ffmpeg = decode any video/animated; pngquant = APNG color optimization;
# apngasm = inter-frame APNG compression (more frames under 512KB); libheif = HEIC;
# gifsicle = lossy-LZW + inter-frame GIF optimization (max-quality export pass).
# oxipng = final lossless APNG deflate squeeze (installed via pip pyoxipng, in-process).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg pngquant apngasm libheif1 gifsicle curl xz-utils \
    && rm -rf /var/lib/apt/lists/*

# gifski = best-in-class GIF encoder (libimagequant per-frame palettes + temporal
# dithering). Not in apt; pull the prebuilt binary tarball and locate the linux binary
# defensively (archive layout has changed across versions). Non-fatal: if it can't be
# fetched, the pipeline degrades to the ffmpeg per-frame-palette path.
ARG GIFSKI_VERSION=1.34.0
RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    if [ "$arch" = "amd64" ]; then \
        ( curl -fsSL "https://gif.ski/gifski-${GIFSKI_VERSION}.tar.xz" -o /tmp/gifski.tar.xz \
          && mkdir -p /tmp/gifski && tar -xf /tmp/gifski.tar.xz -C /tmp/gifski \
          && bin="$(find /tmp/gifski -type f -name gifski | head -n1)" \
          && test -n "$bin" \
          && install -m 0755 "$bin" /usr/local/bin/gifski \
          && rm -rf /tmp/gifski* ) || echo "gifski install skipped (offline or layout changed)"; \
    fi; \
    gifski --version || echo "gifski not available; using ffmpeg GIF fallback"

WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

COPY backend/app ./app
COPY --from=frontend /fe/dist ./frontend/dist

# Bake the default background-removal model into the image so the first
# request isn't slowed by a download. Non-fatal if it can't fetch at build.
RUN python -c "from app.pipeline import bg_removal; bg_removal.warmup([bg_removal.DEFAULT_MODEL])" || true

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
