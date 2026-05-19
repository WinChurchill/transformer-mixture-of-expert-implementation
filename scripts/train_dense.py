from __future__ import annotations

import sys

from train_moe import main


if __name__ == "__main__":
    if "--ffn" not in sys.argv:
        sys.argv.extend(["--ffn", "dense"])
    main()
