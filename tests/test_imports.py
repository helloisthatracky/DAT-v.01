"""Import hygiene.

Guards the architectural rule from the tech spec, p. 6.1: `preview_core` is a
self-contained library that Electron or FastAPI import from the outside. It
must not care what the current working directory is.

Three layers, because the obvious runtime test alone is not enough:

1. Every submodule imports cleanly.
2. Static check that nothing inside the package uses an absolute
   intra-package import. This is the layer that matters - a runtime test
   passes by accident whenever the cwd happens to be `preview_core/`, which
   puts the sibling modules on sys.path and makes `from imaging import ...`
   resolve. That is precisely how such a bug survives review and then fails
   on the first machine that runs the CLI from anywhere else.
3. Import from a foreign directory in a fresh interpreter.
"""

from __future__ import annotations

import ast
import importlib
import os
import pkgutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Located by path, not by import. The static check has to run - and produce a
# readable diagnosis - precisely in the case where importing the package is
# what fails.
ROOT = Path(__file__).resolve().parent.parent
PKG_NAME = "preview_core"
PKG_DIR = ROOT / PKG_NAME

LOCAL_MODULES = {p.stem for p in PKG_DIR.glob("*.py")} - {"__init__", "__main__"}


def test_all_submodules_importable():
    pkg = importlib.import_module(PKG_NAME)
    for mod in pkgutil.walk_packages(pkg.__path__, PKG_NAME + "."):
        importlib.import_module(mod.name)


def test_no_absolute_intra_package_imports():
    offenders: list[str] = []
    for path in sorted(PKG_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                head = (node.module or "").split(".")[0]
                if node.level == 0 and head in LOCAL_MODULES:
                    offenders.append(
                        f"{path.name}:{node.lineno}  from {node.module} import ... "
                        f"-> should be `from .{node.module} import ...`"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in LOCAL_MODULES:
                        offenders.append(
                            f"{path.name}:{node.lineno}  import {alias.name} "
                            f"-> should be `from . import {alias.name}`"
                        )
    assert not offenders, "absolute intra-package imports:\n  " + "\n  ".join(offenders)


def test_package_imports_from_an_unrelated_directory():
    """The exact failure mode: a fresh interpreter, cwd nowhere near the repo."""
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-c", "import preview_core; print(preview_core.__file__)"],
            cwd=tmp,
            capture_output=True,
            text=True,
            env={**_env_with_root()},
        )
    assert result.returncode == 0, result.stderr
    assert "preview_core" in result.stdout


def test_cli_runs_as_a_module_from_an_unrelated_directory():
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-m", "preview_core", "--help"],
            cwd=tmp,
            capture_output=True,
            text=True,
            env={**_env_with_root()},
        )
    assert result.returncode == 0, result.stderr
    assert "build" in result.stdout


def test_default_spec_ships_with_the_package():
    """`--spec` is optional, so the default must resolve in an installed copy."""
    from preview_core.spec import Spec, default_spec_path

    path = default_spec_path()
    assert path.exists(), path
    assert path.parent.name == PKG_NAME, "default spec must live inside the package"
    assert Spec.default().canvas.width > 0


def _env_with_root() -> dict:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    return env
