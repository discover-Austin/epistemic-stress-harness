#!/usr/bin/env python3
"""Run every test_* function in tests/test_*.py without needing pytest.

    python tests/run_all.py            all tests
    python tests/run_all.py -k stats   only modules/functions whose name contains 'stats'

Also pytest-compatible: `pytest -q` from the repo root.
"""

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main(argv) -> int:
    needle = argv[argv.index("-k") + 1] if "-k" in argv else ""
    passed = failed = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            if needle and needle not in name and needle not in path.stem:
                continue
            try:
                getattr(mod, name)()
                passed += 1
                print(f"  PASS  {path.stem}::{name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {path.stem}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
