"""Accept any input (uploaded bytes or a URL), detect its type, fetch safely."""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from ..observability import get_logger

log = get_logger("ingest")

MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024  # hard cap on remote (URL) input
MAX_UPLOAD_BYTES = 50 * 1024 * 1024     # hard cap on direct uploads
MAX_REDIRECTS = 5


class IngestError(Exception):
    """Raised for unsupported / disallowed input."""


class InputKind:
    IMAGE = "image"   # static or animated — decode decides
    VIDEO = "video"


@dataclass
class Source:
    data: bytes
    mime: str
    kind: str
    origin: str  # "upload" | "url"
    filename: str | None = None


def sniff_mime(data: bytes) -> str:
    head = data[:32]
    window = data[: 1 << 16]
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/apng" if b"acTL" in window else "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head[4:12] in (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1", b"ftypmsf1", b"ftypheis"):
        return "image/heic"
    if head[4:8] == b"ftyp":
        return "video/mp4"
    if head[:4] == b"\x1aE\xdf\xa3":
        return "video/webm"  # matroska / webm
    if head[:2] == b"BM":
        return "image/bmp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    return "application/octet-stream"


def kind_for_mime(mime: str) -> str:
    if mime.startswith("video/"):
        return InputKind.VIDEO
    if mime.startswith("image/"):
        return InputKind.IMAGE
    raise IngestError(f"Unsupported input type: {mime}")


def from_bytes(data: bytes, filename: str | None = None) -> Source:
    if not data:
        raise IngestError("Empty input")
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(
            f"File too large ({len(data) // (1024 * 1024)} MB) — the max is "
            f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Try a shorter or lower-resolution GIF."
        )
    mime = sniff_mime(data)
    kind = kind_for_mime(mime)
    log.info("ingest.upload", mime=mime, kind=kind, bytes=len(data), filename=filename)
    return Source(data=data, mime=mime, kind=kind, origin="upload", filename=filename)


def _host_is_public(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def from_url(url: str) -> Source:
    """Fetch a URL with per-hop SSRF validation (blocks private/loopback hosts)."""
    current = url
    with httpx.Client(follow_redirects=False, timeout=20.0) as client:
        for _ in range(MAX_REDIRECTS + 1):
            parsed = urlparse(current)
            if parsed.scheme not in ("http", "https"):
                raise IngestError("Only http/https URLs are allowed")
            if not parsed.hostname or not _host_is_public(parsed.hostname):
                raise IngestError("URL host is not allowed")

            resp = client.get(current, headers={"User-Agent": "discord-sticker-maker/1.0"})
            if resp.is_redirect and "location" in resp.headers:
                current = str(httpx.URL(current).join(resp.headers["location"]))
                log.info("ingest.redirect", to=current)
                continue

            resp.raise_for_status()
            clen = resp.headers.get("content-length")
            if clen and int(clen) > MAX_DOWNLOAD_BYTES:
                raise IngestError("Remote file too large")
            data = resp.content
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise IngestError("Remote file too large")
            mime = sniff_mime(data)
            name = parsed.path.rsplit("/", 1)[-1] or None
            log.info("ingest.url", url=current, mime=mime, bytes=len(data))
            return Source(data=data, mime=mime, kind=kind_for_mime(mime), origin="url", filename=name)

    raise IngestError("Too many redirects")
