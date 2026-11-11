#!/usr/bin/env python3
"""Minimal Mac trader: start workflows at 18:30 & 21:00 IST, run live picks at 9:46."""

import subprocess
import time
from datetime import datetime, timezone, timedelta

IST_OFFSET = timedelta(hours=5, minutes=30)  # IST = UTC+5:30

def now_ist():
    utc = datetime.now(timezone.utc)
    return (utc + IST_OFFSET).time()

def start_workflow(name):
    subprocess.run(["gh", "workflow", "run", name], capture_output=True)
    h, m = now_ist().hour, now_ist().minute
    print(f"{h:02d}:{m:02d} started {name}")

def check_schedule():
    now = now_ist()
    h_m = f"{now.hour:02d}:{now.minute:02d}"
    
    if h_m == "18:30":
        start_workflow("train-models.yml")
    elif h_m == "21:00":
        start_workflow("train-models.yml")
    elif h_m == "09:46" and datetime.now().weekday() < 5:  # Mon-Fri
        print(f"{h_m} live picks would run here (placeholder)")

if __name__ == "__main__":
    last_min = -1
    while True:
        now = now_ist()
        if now.minute != last_min:
            check_schedule()
            last_min = now.minute
        time.sleep(30)
