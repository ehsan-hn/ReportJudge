"""CLI module entry point for eval suite.

Allows execution via:
    python -m eval --provider fake
"""

import asyncio
import sys

from eval.runner import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
