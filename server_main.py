from __future__ import annotations

import sys
from pathlib import Path


SERVER_DIR = Path(__file__).resolve().parent / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from app import main as run_server


def main() -> None:
    run_server()
