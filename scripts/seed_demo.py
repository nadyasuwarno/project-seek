"""Populate the local archive with the synthetic demo systems.

    python scripts/seed_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seek.seed import seed_demo  # noqa: E402

if __name__ == "__main__":
    n = seed_demo(lambda frac, msg: print(f"[{frac:4.0%}] {msg}"))
    print(f"Added {n} demo observations.")
