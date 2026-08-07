"""Find out which copy of `preview_core` Python actually loads, and why.

    python tools/doctor.py

Deliberately imports nothing from the package and depends on nothing outside
the standard library: it has to work in exactly the situation where importing
`preview_core` is what fails.

Answers the question "I fixed the file, why is the error still there?" - which
almost always means the interpreter is reading a different file than the one
that was edited.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import struct
import site
import sys
import sysconfig
from pathlib import Path
from typing import Iterable, List, Tuple

PKG = "preview_core"


# --------------------------------------------------------------------------
def hdr(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def resolve_package() -> Path | None:
    """Where `import preview_core` would read from - without executing it."""
    try:
        spec = importlib.util.find_spec(PKG)
    except Exception as exc:  # noqa: BLE001 - a broken package must not stop us
        print(f"  find_spec raised: {type(exc).__name__}: {exc}")
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def scan_sys_path() -> List[Path]:
    """Every importable copy on sys.path, in resolution order."""
    found: List[Path] = []
    for entry in sys.path:
        if not entry:
            entry = os.getcwd()
        cand = Path(entry) / PKG / "__init__.py"
        try:
            if cand.exists():
                found.append(cand.resolve().parent)
        except OSError:
            continue
    return found


def bad_imports(pkg_dir: Path) -> List[Tuple[str, int, str]]:
    """Absolute intra-package imports - the direct cause of this error."""
    local = {p.stem for p in pkg_dir.glob("*.py")} - {"__init__", "__main__"}
    out: List[Tuple[str, int, str]] = []
    for path in sorted(pkg_dir.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            out.append((path.name, 0, f"cannot parse: {exc}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                head = (node.module or "").split(".")[0]
                if head in local:
                    out.append(
                        (path.name, node.lineno, f"from {node.module} import ...  ->  from .{node.module} import ...")
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in local:
                        out.append(
                            (path.name, node.lineno, f"import {alias.name}  ->  from . import {alias.name}")
                        )
    return out


def stale_bytecode(pkg_dir: Path) -> List[str]:
    """Report cached bytecode that no longer matches its source.

    A .pyc is always newer than its .py - that is normal, not staleness.
    Real staleness is decided by the header: bytes 8..16 hold the source
    mtime and size CPython recorded at compile time. When those disagree with
    the file on disk, CPython recompiles, so a mismatch is informational.
    What is worth naming is a .pyc whose source is gone: usually a module
    that was renamed, with the old name still lying around.
    """
    cache = pkg_dir / "__pycache__"
    if not cache.is_dir():
        return []
    issues: List[str] = []
    rebuilt: List[str] = []
    for pyc in sorted(cache.glob("*.pyc")):
        stem = pyc.name.split(".")[0]
        src = pkg_dir / f"{stem}.py"
        if not src.exists():
            issues.append(f"{pyc.name}: orphan - no {stem}.py beside it (renamed module?)")
            continue
        try:
            head = pyc.read_bytes()[:16]
            if len(head) < 16:
                continue
            flags = struct.unpack("<I", head[4:8])[0]
            if flags & 0b1:  # hash-based pyc, self-validating
                continue
            mtime, size = struct.unpack("<II", head[8:16])
        except OSError:
            continue
        st = src.stat()
        if mtime != int(st.st_mtime) & 0xFFFFFFFF or size != st.st_size & 0xFFFFFFFF:
            rebuilt.append(pyc.name)
    if rebuilt:
        # Normal after copying or syncing a folder; one line, not one per file.
        issues.append(f"{len(rebuilt)} cached files will be recompiled - normal, not a fault")
    return issues


def install_records() -> Iterable[str]:
    """Where pip thinks the package lives."""
    roots: List[Path] = []
    for fn in (site.getsitepackages, lambda: [site.getusersitepackages()]):
        try:
            roots += [Path(p) for p in fn()]
        except Exception:  # noqa: BLE001
            pass
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        roots.append(Path(purelib))

    seen: set[Path] = set()
    for root in roots:
        root = root.resolve() if root.exists() else root
        if root in seen or not root.is_dir():
            continue
        seen.add(root)
        for item in sorted(root.iterdir()):
            name = item.name.lower()
            if PKG.replace("_", "") in name.replace("_", "").replace("-", ""):
                yield f"{item}"
                if item.suffix == ".pth" or "finder" in name:
                    try:
                        for line in item.read_text(encoding="utf-8").splitlines():
                            if line.strip() and PKG in line:
                                yield f"    -> {line.strip()[:160]}"
                    except OSError:
                        pass


# --------------------------------------------------------------------------
def main() -> int:
    print("preview_core doctor")
    print("=" * 60)

    hdr("Interpreter")
    print(f"  executable : {sys.executable}")
    print(f"  version    : {sys.version.split()[0]}")
    print(f"  cwd        : {Path.cwd()}")

    hdr("Project layout")
    root = Path(__file__).resolve().parent.parent
    proj = root / "pyproject.toml"
    print(f"  project root : {root}")
    print(f"  pyproject.toml: {'ok' if proj.exists() else 'MISSING'}")
    if proj.exists():
        print(f"  run installs from here:  cd {root}  &&  pip install -e .")
    else:
        # An archive unpacked one level too deep is the usual reason pip says
        # "does not appear to be a Python project".
        found = list(root.rglob("pyproject.toml"))[:3]
        for f in found:
            print(f"  found instead: {f}  -> install from {f.parent}")
        if not found:
            print("  no pyproject.toml anywhere below the project root")
    nested = root / root.name
    if nested.is_dir() and (nested / "pyproject.toml").exists():
        print(f"  WARNING: nested duplicate at {nested} - the archive was unpacked one level too deep")

    hdr("Which copy would be imported")
    active = resolve_package()
    added_root = False
    if active is None:
        # This script lives in tools/, so the project root is not on sys.path
        # by default. Add it and say so, rather than reporting a false absence.
        root = Path(__file__).resolve().parent.parent
        if (root / PKG / "__init__.py").exists():
            sys.path.insert(0, str(root))
            importlib.invalidate_caches()
            active = resolve_package()
            added_root = True
    if active is None:
        print(f"  {PKG} is NOT importable.")
        print("  Run from the project root, or: pip install -e .")
    else:
        print(f"  {active}")
        if added_root:
            print(f"  (not installed in this environment - found by adding {Path(__file__).resolve().parent.parent})")

    copies = scan_sys_path()
    hdr(f"All copies on sys.path ({len(copies)})")
    if not copies:
        print("  none")
    for i, c in enumerate(copies):
        mark = "  <-- ACTIVE" if active and c == active else ""
        print(f"  [{i}] {c}{mark}")
    if len(copies) > 1:
        print("\n  More than one copy is visible. Python takes the first;")
        print("  edits to any of the others have no effect.")

    hdr("The copy you are probably editing")
    editable: List[Path] = []
    for cand in (Path(__file__).resolve().parent.parent / PKG, Path.cwd() / PKG):
        if (cand / "__init__.py").exists() and cand.resolve() not in editable:
            editable.append(cand.resolve())
    if not editable:
        print("  no preview_core source folder next to this script or in cwd")
    for c in editable:
        if active and c == active:
            print(f"  {c}   <-- this is the one being imported, good")
        else:
            print(f"  {c}")
            print("      NOT the copy Python imports. Edits here have no effect.")
            if active:
                print(f"      Python imports: {active}")

    hdr("pip / site-packages records")
    records = list(install_records())
    if records:
        for r in records:
            print(f"  {r}")
    else:
        print("  no installed record found (package is used from source only)")

    verdict_ok = True

    hdr("Absolute intra-package imports")
    targets = list(dict.fromkeys(copies + editable + ([active] if active else [])))
    if not targets:
        print("  nothing to check")
    for c in targets:
        problems = bad_imports(c)
        tag = "ACTIVE " if active and c == active else ""
        if problems:
            verdict_ok = False
            print(f"  {tag}{c}")
            for fname, line, msg in problems:
                print(f"      {fname}:{line}   {msg}")
        else:
            print(f"  {tag}{c}   clean")

    hdr("Bytecode cache")
    any_stale = False
    for c in targets:
        for issue in stale_bytecode(c):
            any_stale = True
            if ": " in issue:
                name, detail = issue.split(": ", 1)
                print(f"  {c / '__pycache__' / name}")
                print(f"      {detail}")
            else:
                print(f"  {c / '__pycache__'}: {issue}")
    if not any_stale:
        print("  clean")

    hdr("Shipped spec")
    if active:
        spec_file = active / "spec.yaml"
        print(f"  {spec_file}   {'ok' if spec_file.exists() else 'MISSING'}")
        if not spec_file.exists():
            verdict_ok = False

    hdr("Verdict")
    mismatch = bool(editable) and bool(active) and active not in editable
    if mismatch:
        verdict_ok = False
        print("  The folder you edit is not the folder Python imports.")
        print("  Reinstall from the folder you edit, or run from its parent:")
        print(f"       cd {editable[0].parent}")
        print("       pip uninstall -y preview-core && pip install -e .")
    elif verdict_ok and active:
        print("  No import problems found in the copy Python loads.")
        print("  If the CLI still fails, the `preview` script points at another")
        print("  environment - compare its interpreter with the one above.")
    else:
        print("  Problems found. In order:")
        print("    1. fix any absolute imports listed above (add the leading dot)")
        print("    2. delete __pycache__ folders inside the package")
        print("    3. reinstall from the directory you actually edit:")
        print("         pip uninstall -y preview-core && pip install -e .")
    return 0 if verdict_ok else 1


if __name__ == "__main__":
    sys.exit(main())
