"""Run the test suite and require 100% statement and branch coverage."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import coverage

# Import torch before coverage tracing. Some builds register C-extension
# docstrings at import time in ways that do not interact well with tracing.
import torch  # noqa: F401


def main() -> int:
    cov = coverage.Coverage(branch=True, source=["rosa"])
    cov.erase()
    cov.start()
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    cov.stop()
    cov.save()
    percent = cov.report(show_missing=True)
    return 0 if result.wasSuccessful() and percent == 100.0 else 1


if __name__ == "__main__":
    sys.exit(main())
