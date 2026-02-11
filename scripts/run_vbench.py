#!/usr/bin/env python3
"""
Thin entrypoint for VBench evaluation.

Implementation lives in `scripts/vbench_runner/`.
"""

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from vbench_runner.core import main


if __name__ == "__main__":
    main()

