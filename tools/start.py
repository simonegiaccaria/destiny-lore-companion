"""One launcher: make sure Ollama is up, then start the server or the eval.

Usage (from the project root, or via avvia.bat):
    .venv\\Scripts\\python tools\\start.py                 # Ollama + web server
    .venv\\Scripts\\python tools\\start.py --port 8080     # different port
    .venv\\Scripts\\python tools\\start.py eval --repeat 3 --temperature 0.2
    .venv\\Scripts\\python tools\\start.py ollama          # Ollama only

Everything after "eval" is passed unchanged to eval/run_eval.py.
Ctrl+C stops the server/eval, and Ollama too if this script started it.
If Ollama was already running (e.g. the tray app), it is left alone.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "data" / "logs"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
OLLAMA_START_TIMEOUT = 30  # seconds


def load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("[start] python-dotenv not installed; .env not read")
        return
    load_dotenv(ROOT / ".env")


def ollama_root() -> str | None:
    """Base URL of the local Ollama server, or None if the LLM is remote."""
    base = os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    parsed = urlparse(base)
    if parsed.hostname not in LOCAL_HOSTS:
        return None
    return f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 11434}"


def http_get_json(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def ollama_up(root: str) -> bool:
    try:
        http_get_json(f"{root}/api/version")
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def find_ollama() -> str | None:
    exe = shutil.which("ollama")
    if exe:
        return exe
    if os.name == "nt":
        default = Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
        if default.is_file():
            return str(default)
    return None


def start_ollama(exe: str) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / "ollama.log", "a", encoding="utf-8")
    kwargs = {"stdout": log, "stderr": subprocess.STDOUT}
    # Own process group: Ctrl+C reaches us only, and we stop Ollama ourselves.
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen([exe, "serve"], **kwargs)


def wait_for_ollama(root: str, proc: subprocess.Popen) -> bool:
    deadline = time.monotonic() + OLLAMA_START_TIMEOUT
    while time.monotonic() < deadline:
        if ollama_up(root):
            return True
        if proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def check_model(root: str) -> None:
    model = os.getenv("LLM_MODEL", "")
    if not model:
        return
    try:
        tags = http_get_json(f"{root}/api/tags", timeout=5)
    except (urllib.error.URLError, OSError, ValueError):
        print("[start] could not list Ollama models; skipping model check")
        return
    names = {m.get("name", "") for m in tags.get("models", [])}
    wanted = model if ":" in model else f"{model}:latest"
    if wanted in names:
        print(f"[start] model present: {model}")
    else:
        print(f"[start] WARNING: model {model!r} not found in Ollama.")
        print(f"        Download it with:  ollama pull {model}")


def ensure_ollama() -> subprocess.Popen | None:
    """Return the Popen if we started Ollama, None if it was already up/remote."""
    root = ollama_root()
    if root is None:
        print("[start] LLM_BASE_URL is not local: Ollama not needed")
        return None
    if ollama_up(root):
        print(f"[start] Ollama already running at {root}")
        check_model(root)
        return None
    exe = find_ollama()
    if exe is None:
        sys.exit("[start] Ollama not found. Install it or add it to PATH.")
    print(f"[start] starting Ollama ({exe}) — log: {LOG_DIR / 'ollama.log'}")
    proc = start_ollama(exe)
    if not wait_for_ollama(root, proc):
        stop(proc, "Ollama")
        sys.exit(f"[start] Ollama did not come up within {OLLAMA_START_TIMEOUT}s; "
                 f"see {LOG_DIR / 'ollama.log'}")
    print(f"[start] Ollama up at {root}")
    check_model(root)
    return proc


def stop(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"[start] stopping {name}")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def lan_ip() -> str:
    """IP of the interface used for outbound traffic (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.1.1", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def run_child(cmd: list[str]) -> int:
    """Run a child in the foreground; Ctrl+C reaches it too (same console)."""
    proc = subprocess.Popen(cmd, cwd=ROOT)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return 1


def wait_forever() -> None:
    print("[start] Ollama only. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="server",
                        choices=["server", "eval", "ollama"])
    parser.add_argument("--host", default="0.0.0.0",
                        help="server bind address (0.0.0.0 = reachable from the phone)")
    parser.add_argument("--port", type=int, default=8000)
    args, extra = parser.parse_known_args()

    if args.mode != "eval" and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    os.chdir(ROOT)
    load_env()
    if sys.prefix == sys.base_prefix:
        print("[start] WARNING: not running inside a virtualenv; "
              "use .venv\\Scripts\\python or avvia.bat")

    ollama_proc = ensure_ollama()
    try:
        if args.mode == "server":
            print(f"[start] server: http://127.0.0.1:{args.port}  "
                  f"| phone: http://{lan_ip()}:{args.port}")
            return run_child([sys.executable, "-m", "uvicorn", "app.server:app",
                              "--host", args.host, "--port", str(args.port)])
        if args.mode == "eval":
            return run_child([sys.executable, str(ROOT / "eval" / "run_eval.py"), *extra])
        wait_forever()
        return 0
    finally:
        stop(ollama_proc, "Ollama")


if __name__ == "__main__":
    sys.exit(main())