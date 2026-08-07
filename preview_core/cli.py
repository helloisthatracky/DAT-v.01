"""CLI: preview-build / preview-validate.

Usage
    preview-build job.json --spec spec.yaml -o out/
    preview-build --views a.png b.png --model-id 12345 -o out/
    preview-validate ready.jpg --role main --spec spec.yaml
"""

from __future__ import annotations

# Relative imports below need package context, which a plain `python path.py`
# does not provide. Fail here with instructions instead of letting Python
# raise "attempted relative import with no known parent package" ten lines on.
if __name__ == "__main__" and __package__ in (None, ""):
    raise SystemExit(
        "preview_core.cli is a package module, not a standalone script.\n"
        "Run it one of these ways from the project root:\n"
        "    python -m preview_core build job.json -o out/\n"
        "    python -m preview_core.cli build job.json -o out/\n"
        "    pip install -e .   &&   preview build job.json -o out/"
    )

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .models import Job, View
from .pipeline import build_previews
from .spec import Spec, default_spec_path
from .validate import validate_file


def _load_spec(path: str | None) -> Spec:
    return Spec.load(path) if path else Spec.default()


def _job_from_args(args: argparse.Namespace) -> Job:
    if args.job:
        base = Path(args.job).resolve().parent
        return Job.load(args.job).resolve(base)
    if not args.views:
        raise SystemExit("give either job.json or --views")
    return Job(
        job_id=args.job_id or "cli",
        model_id=args.model_id or "model",
        views=[View(url=v) for v in args.views],
        spheres=list(args.spheres or []),
        brand_logo=args.brand_logo,
        outputs=args.outputs.split(","),
    )


def cmd_build(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    job = _job_from_args(args)
    report = build_previews(job, spec, args.out, logo_3ddd_path=args.logo_3ddd)

    out = Path(args.out)
    report.save(out / "report.json")
    (out / "job.json").write_text(
        json.dumps(
            {
                "job_id": job.job_id,
                "model_id": job.model_id,
                "views": [v.__dict__ for v in job.views],
                "spheres": job.spheres,
                "brand_logo": job.brand_logo,
                "outputs": job.outputs,
                "overrides": job.overrides,
                "spec_version": spec.version,
                "spec_digest": spec.digest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"spec: {spec.source_path}  (v{spec.version}, {spec.digest})")
    _print_report(report)
    return 1 if report.status == "failed" else 0


def cmd_validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.spec)
    worst = 0
    for f in args.files:
        checks = validate_file(f, args.role, spec)
        bad = [c for c in checks if c.status != "pass"]
        mark = "FAIL" if any(c.status == "fail" for c in checks) else ("WARN" if bad else "PASS")
        print(f"{mark:4}  {f}")
        for c in bad:
            print(f"        {c.status:4} {c.id}: {c.msg}")
        worst = max(worst, 2 if mark == "FAIL" else (1 if mark == "WARN" else 0))
    return 1 if worst == 2 else 0


def _print_report(report) -> None:  # noqa: ANN001 - internal pretty printer
    d = report.detected
    print(
        f"type={d.get('type')} ratio={d.get('ratio')} views={d.get('views')} "
        f"in {d.get('elapsed_s')}s  ->  {report.status}"
    )
    for r in report.results:
        print(f"  #{r.index} {r.role:10} {r.status:5} {Path(r.path).name}")
        for c in r.checks:
            if c.status != "pass":
                print(f"      {c.status:4} {c.id}: {c.msg}")
        for w in r.warnings:
            print(f"      note  {w}")
    for w in report.warnings:
        print(f"  note  {w}")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="preview", description="3ddd preview builder")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a preview set")
    b.add_argument("job", nargs="?", help="job.json")
    b.add_argument("--views", nargs="+")
    b.add_argument("--spheres", nargs="*")
    b.add_argument("--brand-logo")
    b.add_argument("--logo-3ddd")
    b.add_argument("--model-id")
    b.add_argument("--job-id")
    b.add_argument("--outputs", default="main,secondary,closeup")
    b.add_argument("--spec", default=None, help=f"default: {default_spec_path()}")
    b.add_argument("-o", "--out", default="out")
    b.set_defaults(func=cmd_build)

    v = sub.add_parser("validate", help="check existing previews against the guide")
    v.add_argument("files", nargs="+")
    v.add_argument("--role", default="main", choices=["main", "secondary", "closeup"])
    v.add_argument("--spec", default=None, help=f"default: {default_spec_path()}")
    v.set_defaults(func=cmd_validate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
