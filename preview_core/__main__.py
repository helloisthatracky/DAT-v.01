"""Entry point for `python -m preview_core`.

Exists so the package is runnable without installing it and without the
`.cli` suffix:

    python -m preview_core build job.json -o out/
"""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
