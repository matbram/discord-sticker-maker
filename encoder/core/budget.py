"""Anytime budget: a wall-clock deadline and/or an attempt (engine-run) cap.

The search calls ``expired()`` at each phase/loop boundary and ``tick()`` after
every engine run. When the budget is hit the search returns its best result so
far, and the report notes that it stopped early and why. The bench uses an
attempt cap (count-based, reproducible); the interactive path uses wall-clock.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    seconds: float | None = None
    max_attempts: int | None = None
    attempts: int = 0
    stop_reason: str | None = None
    _start: float = field(default_factory=time.monotonic)

    def tick(self) -> None:
        self.attempts += 1

    def expired(self) -> bool:
        if self.max_attempts is not None and self.attempts >= self.max_attempts:
            self.stop_reason = self.stop_reason or "max_attempts"
            return True
        if self.seconds is not None and (time.monotonic() - self._start) >= self.seconds:
            self.stop_reason = self.stop_reason or "budget_seconds"
            return True
        return False

    @property
    def stopped_early(self) -> bool:
        return self.stop_reason in ("max_attempts", "budget_seconds")

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000.0
