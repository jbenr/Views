"""Root-level data refresh launcher."""

import subprocess
import sys
from pathlib import Path


def test_dotted_ref_launcher_forwards_cli_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "data_pull.ref"), "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0
    assert "--skip-bbg" in result.stdout
    assert "--only" in result.stdout
