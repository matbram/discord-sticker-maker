"""``encode()`` — the public entry point that turns a source + byte target into a
spec-compliant GIF plus an honesty report.

It wires the pieces together: decode -> choose engine -> run the anytime guided
search (driving the engine for real measured sizes, judging fitting candidates
with the perceptual metric) -> copy out the winner -> assemble the report. Every
frame is preserved; size comes from palette/dither/lossy/resolution levers only.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import numpy as np

from ..metrics import default_metric
from ..metrics.base import Metric
from .budget import Budget
from .engines import Engine, available_engines, prepare_context
from .frames import Frames, InputCaps, frames_from_list, frames_from_source, load_gif, sniff_kind
from .levers import SCALE_VALUES, LeverState
from .logging import get_logger
from .result import EncodeResult, LossLocus
from .search import Candidate, SearchOutcome, guided_search
from .sizes import Tolerance, parse_tolerance, preset_bytes
from .timing import CLAMP_WARN_FPS, centiseconds_to_ms, effective_fps

log = get_logger("encode")

_NO_CEILING = 1 << 50   # effectively unbounded for "invisible" with no target


class NoEngineError(RuntimeError):
    pass


def _has_alpha(frames: Frames) -> bool:
    return any(fr.shape[-1] == 4 and bool(np.any(fr[..., 3] < 255)) for fr in frames.frames)


def _select_engine(frames: Frames, names: list[str] | None) -> tuple[Engine, list[str]]:
    engines = {e.name: e for e in available_engines(names)}
    if not engines:
        raise NoEngineError(
            "no GIF engine available — install ffmpeg, gifski, or gifsicle"
        )
    warnings: list[str] = []
    alpha = _has_alpha(frames)
    preference = (
        ["ffmpeg-palette", "gifsicle-lossy", "gifski"] if alpha
        else ["ffmpeg-palette", "gifski", "gifsicle-lossy"]
    )
    for name in preference:
        if name in engines:
            eng = engines[name]
            if alpha and not eng.supports_alpha():
                warnings.append("transparency flattened to 1-bit (engine has no partial alpha)")
            return eng, warnings
    eng = next(iter(engines.values()))
    return eng, warnings


def _region_hint(frames: Frames, worst_idx: int) -> str:
    """A coarse, human-readable guess at where loss is most visible."""
    return "fast motion or fine detail near frame %d" % worst_idx


def encode(
    source,
    target_bytes: int | None,
    mode: str = "cap",
    *,
    delays_ms=None,
    fps: float | None = None,
    max_fps: float = 50.0,
    platform: str | None = None,
    tolerance=None,
    budget_seconds: float = 30.0,
    max_attempts: int = 24,
    metric: Metric | None = None,
    caps: InputCaps | None = None,
    engines: list[str] | None = None,
    out_path: str | None = None,
    report_path: str | None = None,
):
    """Encode ``source`` to a GIF that fits ``target_bytes`` (or is invisibly small).

    ``source`` is a path (video/gif/image) or a list of RGBA frame arrays. Returns
    an :class:`EncodeResult`; when ``report_path`` is given (or the CLI sets it) the
    full :class:`EncodeReport` JSON is written there.
    """
    if mode not in ("cap", "invisible"):
        raise ValueError(f"mode must be 'cap' or 'invisible', got {mode!r}")

    # ---- resolve the byte target -------------------------------------------------
    if target_bytes is None and platform:
        target_bytes = preset_bytes(platform)
    if target_bytes is None and mode == "cap":
        raise ValueError("cap mode requires target_bytes or a platform preset")
    ceiling = target_bytes if target_bytes is not None else _NO_CEILING

    tol = parse_tolerance(tolerance) if tolerance is not None else Tolerance()
    metric = metric or default_metric()

    # ---- decode to frames (every frame kept) ------------------------------------
    if isinstance(source, (list, tuple)):
        frames = frames_from_list(list(source), delays_ms)
        input_kind = "frames"
        input_path = "<frames>"
    else:
        input_path = str(source)
        input_kind = sniff_kind(input_path)
        frames = frames_from_source(input_path, fps=fps, max_fps=max_fps, caps=caps)

    engine, warnings = _select_engine(frames, engines)
    log.info("encode.start", kind=input_kind, frames=frames.n, engine=engine.name,
             mode=mode, target=target_bytes)

    budget = Budget(seconds=budget_seconds, max_attempts=max_attempts)
    workdir = tempfile.mkdtemp(prefix="fovea_")
    contexts: dict[float, object] = {}

    def ctx_for(scale: float):
        if scale not in contexts:
            d = os.path.join(workdir, f"s{int(round(scale * 100))}")
            os.makedirs(d, exist_ok=True)
            contexts[scale] = prepare_context(frames, scale, d)
        return contexts[scale]

    def measure(scale: float, idx: int) -> Candidate:
        ctx = ctx_for(scale)
        state = engine.state_for_primary(idx).with_(scale=scale)
        out = os.path.join(workdir, f"c_{int(round(scale * 100))}_{idx}.gif")
        eo = engine.encode(ctx, state, out)
        return Candidate(idx=idx, size_bytes=eo.size_bytes, state=eo.state, out_path=out, scale=scale)

    def score(cand: Candidate) -> float:
        if cand.result is None:
            cand.result = metric.distance(frames, load_gif(cand.out_path))
        return cand.result.distance

    def explore(anchor: Candidate, bud: Budget) -> list[Candidate]:
        out_list: list[Candidate] = []
        for j, nstate in enumerate(engine.secondary_neighbors(anchor.state)):
            if bud.expired():
                break
            out = os.path.join(workdir, f"n_{int(round(anchor.scale * 100))}_{anchor.idx}_{j}.gif")
            try:
                eo = engine.encode(ctx_for(anchor.scale), nstate.with_(scale=anchor.scale), out)
            except Exception as exc:  # noqa: BLE001 - a bad lever combo shouldn't abort
                log.warning("encode.neighbor_failed", err=str(exc)[:120])
                bud.tick()
                continue
            bud.tick()
            out_list.append(Candidate(anchor.idx, eo.size_bytes, eo.state, out, anchor.scale))
        return out_list

    try:
        outcome = guided_search(
            primary_n=len(engine.primary_values()),
            scales=list(SCALE_VALUES),
            measure=measure,
            score=score,
            target_bytes=ceiling,
            tol=tol,
            budget=budget,
            mode=mode,
            invisible_threshold=metric.invisible_threshold,
            explore=explore,
        )
        result = _finalize(
            outcome, frames, engine, metric, mode, target_bytes, input_path,
            input_kind, ctx_for, warnings, budget, out_path, report_path,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return result


def _finalize(
    outcome: SearchOutcome, frames: Frames, engine: Engine, metric: Metric, mode: str,
    target_bytes: int | None, input_path: str, input_kind: str, ctx_for, warnings: list[str],
    budget: Budget, out_path: str | None, report_path: str | None,
) -> EncodeResult:
    from .result import EncodeReport

    chosen = outcome.chosen
    if chosen is None:
        raise RuntimeError("encode produced no candidate (no engine output)")
    if chosen.result is None:
        chosen.result = metric.distance(frames, load_gif(chosen.out_path))

    final_path = out_path or os.path.join(tempfile.gettempdir(), "fovea_out.gif")
    shutil.copyfile(chosen.out_path, final_path)

    ctx = ctx_for(chosen.scale)
    delays_ms = centiseconds_to_ms(ctx.delays_cs)
    out_fps = effective_fps(delays_ms)
    duration_ms = int(sum(delays_ms))
    distance = chosen.result.distance
    lossless = distance <= metric.invisible_threshold
    under_target = (target_bytes is None) or (chosen.size_bytes <= target_bytes)

    notes: list[str] = list(warnings)
    if not frames.delays_ms or len(set(frames.delays_ms)) > 1:
        notes.append(f"per-frame timing mapped to a constant ~{out_fps} fps")
    if out_fps and out_fps > CLAMP_WARN_FPS:
        notes.append(f"effective {out_fps} fps may be clamped by some players toward ~10 fps")
    if not under_target:
        notes.append("could not fit the target even at lowest resolution; smallest result returned")
    if outcome.stopped_early:
        notes.append(f"search stopped early ({outcome.stop_reason}); returned best-so-far")

    loss = None
    if not lossless:
        loss = LossLocus(
            worst_frame=chosen.result.worst_frame,
            worst_frame_distance=(chosen.result.per_frame[chosen.result.worst_frame]
                                  if chosen.result.per_frame else distance),
            region_hint=_region_hint(frames, chosen.result.worst_frame),
        )

    report = EncodeReport(
        input_path=input_path,
        input_kind=input_kind,
        mode=mode,
        target_bytes=target_bytes,
        achieved_bytes=chosen.size_bytes,
        under_target=under_target,
        perceptually_lossless=lossless,
        perceptual_distance=round(distance, 6),
        metric_name=metric.name,
        invisible_threshold=metric.invisible_threshold,
        output_fps=out_fps,
        n_frames=frames.n,
        duration_ms=duration_ms,
        engine_used=engine.name,
        lever_setting=chosen.state.as_dict(),
        loss_locus=loss,
        stopped_early=outcome.stopped_early,
        stop_reason=outcome.stop_reason,
        attempts=budget.attempts,
        elapsed_ms=round(budget.elapsed_ms(), 1),
        warnings=notes,
    )
    if report_path:
        with open(report_path, "w") as fh:
            fh.write(report.model_dump_json(indent=2))

    log.info("encode.done", bytes=chosen.size_bytes, lossless=lossless, fps=out_fps,
             engine=engine.name, attempts=budget.attempts)
    return EncodeResult(
        path=final_path,
        size_bytes=chosen.size_bytes,
        perceptually_lossless=lossless,
        output_fps=out_fps,
        notes=notes,
    )
