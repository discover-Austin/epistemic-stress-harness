#!/usr/bin/env python3
"""
Legacy shim. v0.1 compared three hard-coded single responses; that cannot
separate an effect from sampling noise.

    python compare.py <dir>        side-by-side view of single-result JSON files
    python cli.py run ... && python cli.py analyze <run_dir>     the real analysis
"""

import sys

from cli import main

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(["compare", sys.argv[1]])
