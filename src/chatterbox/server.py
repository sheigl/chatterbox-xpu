"""Console-script launcher for the OpenAI-compatible API server.

`openai_server.py` lives at the repository root (outside the installed
package), so this thin module puts the repo root on ``sys.path`` before
delegating to it.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import openai_server

    openai_server.main()