#!/usr/bin/env python
"""run data ingestion pipeline."""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.ingestion import main  # noqa: E402

if __name__ == "__main__":
    main()
