"""`fovea-bench` CLI: run | validate | list."""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MANIFEST = os.path.join("bench", "corpus", "manifest.yaml")
DEFAULT_CORPUS = os.path.join("bench", "corpus")


def _add_common(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--manifest", default=DEFAULT_MANIFEST)
    sp.add_argument("--corpus-dir", default=DEFAULT_CORPUS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fovea-bench", description="Fovea benchmark harness (M0).")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run baselines (and optionally fovea) over the corpus")
    _add_common(run)
    run.add_argument("--engines", default=None,
                     help="comma list (default: the baselines; add 'fovea' to compare)")
    run.add_argument("--out-dir", default=os.path.join("bench", "out"))
    run.add_argument("--max-attempts", type=int, default=12)
    run.add_argument("--metric", default="auto")
    run.add_argument("--fps", type=float, default=None)

    val = sub.add_parser("validate", help="schema-check the manifest; report clip/binary status")
    _add_common(val)

    lst = sub.add_parser("list", help="print the clips x targets that would run")
    _add_common(lst)
    return p


def _cmd_run(args) -> int:
    from .run import format_run_summary, run_bench

    engines = [e.strip() for e in args.engines.split(",")] if args.engines else None
    records, meta = run_bench(
        args.manifest, args.corpus_dir, engine_names=engines, out_dir=args.out_dir,
        max_attempts=args.max_attempts, metric_name=args.metric, fps=args.fps,
    )
    print(format_run_summary(records, meta))
    print(f"\nwrote: {os.path.join(args.out_dir, 'results.csv')} and results.json")
    return 0


def _cmd_validate(args) -> int:
    from encoder.core.engines import ALL_ENGINES

    from .manifest import clip_present, load_manifest

    try:
        manifest = load_manifest(args.manifest)
    except Exception as exc:  # noqa: BLE001
        print(f"fovea-bench: invalid manifest: {exc}", file=sys.stderr)
        return 1
    present = sum(clip_present(c, args.corpus_dir) for c in manifest.clips)
    print(f"manifest OK: version={manifest.version}, clips={len(manifest.clips)}, "
          f"present={present}/{len(manifest.clips)}")
    for cls in ALL_ENGINES:
        print(f"  engine {cls.name:<16} {'available' if cls.available() else 'MISSING'}")
    if present == 0:
        print("note: no clips present — drop real clips into "
              f"{os.path.join(args.corpus_dir, 'clips')}/ to run the benchmark")
    return 0


def _cmd_list(args) -> int:
    from .manifest import clip_present, load_manifest, resolved_targets

    manifest = load_manifest(args.manifest)
    for clip in manifest.clips:
        flag = "present" if clip_present(clip, args.corpus_dir) else "missing"
        targets = ", ".join(str(t) for t in resolved_targets(clip, manifest))
        print(f"{clip.id} [{clip.category}] ({flag})  targets: {targets}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return {"run": _cmd_run, "validate": _cmd_validate, "list": _cmd_list}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
