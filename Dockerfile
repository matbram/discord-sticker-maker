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
# gifsicle = a Fovea encoder engine (lossy-LZW GIF post-pass). gifski is optional
# (no apt package); add it via a release binary or `cargo install gifski` if the
# opaque video->GIF path is needed — Fovea degrades gracefully without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg pngquant apngasm libheif1 gifsicle \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt

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
