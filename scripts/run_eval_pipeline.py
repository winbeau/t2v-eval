#!/usr/bin/env python3
"""
run_eval_pipeline.py (legacy alias)

Deprecated: use `scripts/run_eval_core.py`.
"""

import warnings


def main():
    warnings.warn(
        "`scripts/run_eval_pipeline.py` is deprecated; use `scripts/run_eval_core.py` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from scripts.run_eval_core import main as pipeline_main
    except ImportError:
        from run_eval_core import main as pipeline_main
    pipeline_main()


if __name__ == "__main__":
    main()
