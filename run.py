#!/usr/bin/env python
"""Entry point. See evalaware/experiments/run_all.py for options."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evalaware.experiments.run_all import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli())
