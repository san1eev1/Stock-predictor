"""Background training job on the Mac (launchd), so the models keep learning every day.

Runs `python -m stockpredictor auto` every day at 18:00 and 21:30 (the second run catches days
when market data is published late). launchd also runs a missed job once when the Mac wakes.
It complements `autostart` (weekday live monitor): it covers weekend self-tuning and any day
the monitor did not run, and does nothing while the monitor is running.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from pathlib import Path

from stockpredictor.config import PROJECT_ROOT

LABEL = "com.stockpredictor.train"   # separate from `autostart` (com.stockpredictor.daily)
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
LOG_PATH = PROJECT_ROOT / "logs" / "auto.log"
RUN_TIMES = [(18, 0), (21, 30)]


def build_plist(python: str = sys.executable, root: Path = PROJECT_ROOT) -> dict:
    return {
        "Label": LABEL,
        # caffeinate keeps the Mac awake until training finishes
        "ProgramArguments": ["/usr/bin/caffeinate", "-i", python, "-m", "stockpredictor", "auto"],
        "WorkingDirectory": str(root),
        "StartCalendarInterval": [{"Hour": h, "Minute": m} for h, m in RUN_TIMES],
        "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"},
        "StandardOutPath": str(LOG_PATH),
        "StandardErrorPath": str(LOG_PATH),
        "ProcessType": "Background",
    }


def _launchctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def install() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLIST_PATH.write_bytes(plistlib.dumps(build_plist()))
    domain = f"gui/{os.getuid()}"
    _launchctl("bootout", f"{domain}/{LABEL}")   # replace an older version, if any
    res = _launchctl("bootstrap", domain, str(PLIST_PATH))
    if res.returncode != 0:
        raise RuntimeError(f"launchctl bootstrap failed: {res.stderr.strip()}")


def remove() -> None:
    _launchctl("bootout", f"gui/{os.getuid()}/{LABEL}")
    PLIST_PATH.unlink(missing_ok=True)


def is_installed() -> bool:
    return _launchctl("print", f"gui/{os.getuid()}/{LABEL}").returncode == 0
