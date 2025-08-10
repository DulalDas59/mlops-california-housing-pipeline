import os, time, subprocess
from pathlib import Path
from threading import Timer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

IGNORE_DIRS = {".git", ".dvc", "models", "metrics", "__pycache__", "venv", ".venv"}

class Handler(FileSystemEventHandler):
    def __init__(self, debounce=0.75):
        self._timer = None
        self._debounce = debounce

    def _trigger(self):
        # Only rerun if DVC thinks something changed
        if subprocess.run("dvc status -q", shell=True).returncode == 0:
            print("[watch] DVC up-to-date; skip repro.")
            return
        print("[watch] running dvc repro...")
        rc = subprocess.run("dvc repro", shell=True).returncode
        print("[watch] done" if rc == 0 else f"[watch] failed: {rc}")

    def _debounced(self):
        if self._timer:
            self._timer.cancel()
        self._timer = Timer(self._debounce, self._trigger)
        self._timer.start()

    def on_any_event(self, event):
        path = event.src_path
        parts = Path(path).parts
        if any(p in IGNORE_DIRS for p in parts):
            return
        # Only react to file-level changes in inputs
        if path.endswith(".py") or path.endswith(".csv") or path.endswith(".yaml") or path.endswith(".yml"):
            self._debounced()

def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    obs = Observer()
    h = Handler()

    # Watch directories instead of single file
    obs.schedule(h, "data/raw", recursive=True)
    obs.schedule(h, "src", recursive=True)
    obs.schedule(h, ".", recursive=False)  # catch params.yaml in repo root
    obs.start()
    print("[watch] watching data/raw/**, src/**, and params.yaml")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        obs.stop()
    obs.join()

if __name__ == "__main__":
    main()
