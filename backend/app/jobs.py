"""In-memory job store. One asyncio queue per job carries SSE progress events."""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    id: str
    request_id: str
    queue: "asyncio.Queue" = field(default_factory=asyncio.Queue)
    status: str = "pending"  # pending | running | done | error
    # outputs keyed by type ("sticker"|"emoji"|"gif") -> {bytes, fmt, meta}
    outputs: dict = field(default_factory=dict)
    order: list = field(default_factory=list)  # output types in request order
    error: Optional[str] = None
    created: float = field(default_factory=time.time)

    @property
    def first(self):
        return self.outputs.get(self.order[0]) if self.order else None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, job_id: str, request_id: str) -> Job:
        self.cleanup()
        job = Job(id=job_id, request_id=request_id)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cleanup(self, ttl_seconds: float = 900) -> None:
        now = time.time()
        stale = [k for k, v in self._jobs.items() if now - v.created > ttl_seconds]
        for k in stale:
            self._jobs.pop(k, None)
