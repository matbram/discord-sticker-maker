"""``fovea`` command-line interface.

    fovea encode INPUT (--target-size 8MB | --platform discord) [--mode cap|invisible]
        [--fps N] [--max-fps 50] [--tolerance 5%] [--budget-seconds 30]
        [--max-attempts 24] [--metric auto] [--engines ...] -o OUT.gif [--report OUT.json]

Emits the GIF, a JSON report sidecar, and a one-line human summary.
"""
from __future__ import annotations

import argparse
import sys


def _human(num: int) -> str:
    f = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if f < 1024 or unit == "GB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024
    return f"{f:.1f}GB"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fovea", description="Perceptually-lossless GIF encoder.")
    sub = p.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="encode a video/GIF to a size-targeted GIF")
    enc.add_argument("input", help="input video or GIF")
    enc.add_argument("--target-size", default=None,
                     help="byte target, e.g. 8MB / 512KB (binary units; never exceeded)")
    enc.add_argument("--platform", default=None, help="preset target (discord, slack, ...)")
    enc.add_argument("--mode", choices=["cap", "invisible"], default="cap")
    enc.add_argument("--fps", type=float, default=None, help="output fps (default: source)")
    enc.add_argument("--max-fps", type=float, default=50.0)
    enc.add_argument("--tolerance", default=None, help="under-target window, e.g. 5%%")
    enc.add_argument("--budget-seconds", type=float, default=30.0)
    enc.add_argument("--max-attempts", type=int, default=24)
    enc.add_argument("--metric", default="auto")
    enc.add_argument("--engines", default=None, help="comma-separated engine allowlist")
    enc.add_argument("-o", "--output", required=True, help="output .gif path")
    enc.add_argument("--report", default=None, help="JSON report path (default: OUT.gif.json)")
    enc.add_argument("-v", "--verbose", action="store_true")
    return p


def _cmd_encode(args: argparse.Namespace) -> int:
    import logging

    from ..core.encode import encode
    from ..core.sizes import parse_size_str
    from ..metrics import get_metric

    if args.verbose:
        logging.getLogger("fovea").setLevel(logging.DEBUG)

    try:
        target = parse_size_str(args.target_size) if args.target_size else None
        metric = get_metric(args.metric)
        report_path = args.report or (args.output + ".json")
        engines = [e.strip() for e in args.engines.split(",")] if args.engines else None
        res = encode(
            args.input, target, args.mode,
            fps=args.fps, max_fps=args.max_fps, platform=args.platform,
            tolerance=args.tolerance, budget_seconds=args.budget_seconds,
            max_attempts=args.max_attempts, metric=metric, engines=engines,
            out_path=args.output, report_path=report_path,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        print(f"fovea: error: {exc}", file=sys.stderr)
        return 1

    status = "perceptually lossless" if res.perceptually_lossless else "VISIBLE trade-off"
    fps = f"{res.output_fps} fps" if res.output_fps else "static"
    print(f"{args.output}  {_human(res.size_bytes)}  [{status}]  {fps}")
    for note in res.notes:
        print(f"  - {note}")
    print(f"report: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "encode":
        return _cmd_encode(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
