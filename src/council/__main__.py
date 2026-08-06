"""``python -m council``.

Thin on purpose: :mod:`council.cli` returns an exit code rather than calling
``sys.exit``, so every subcommand is callable from a test without a process.
"""

from __future__ import annotations

import sys

from council.cli import main

if __name__ == "__main__":
    sys.exit(main())
