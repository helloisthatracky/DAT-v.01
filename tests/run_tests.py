"""Minimal test runner.

pytest is the intended way to run these (`pytest -q`). This fallback exists
so the suite is runnable on a bare machine with nothing but numpy and PIL -
which is exactly the machine a new developer starts on.

    python tests/run_tests.py
"""

from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULES = [
    "tests.test_imports",
    "tests.test_geometry",
    "tests.test_layout",
    "tests.test_validate",
    "tests.test_pipeline",
]


def main() -> int:
    passed: list[str] = []
    failed: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - a broken module is a result, not a crash
            failed.append((f"{name} (collection)", traceback.format_exc()))
            print(f"  FAIL  {name} (collection)")
            continue
        for attr in sorted(vars(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            label = f"{name.split('.')[-1]}::{attr}"
            try:
                fn()
                passed.append(label)
                print(f"  ok    {label}")
            except Exception:  # noqa: BLE001 - a runner reports, it does not judge
                failed.append((label, traceback.format_exc()))
                print(f"  FAIL  {label}")

    print()
    for label, tb in failed:
        print(f"--- {label} " + "-" * (60 - len(label)))
        print(tb)

    dt = time.perf_counter() - t0
    print(f"{len(passed)} passed, {len(failed)} failed in {dt:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
