"""Start the local API and UI using the current Python. Never install or download anything."""

import argparse
import importlib.util
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-port", type=int, default=8020)
    parser.add_argument("--ui-port", type=int, default=8502)
    args = parser.parse_args()
    missing = [
        p
        for p in ("fastapi", "streamlit", "uvicorn", "httpx", "numpy")
        if importlib.util.find_spec(p) is None
    ]
    if missing:
        raise SystemExit("Missing Python packages: " + ", ".join(missing) + ". See README setup.")
    for port in (args.api_port, args.ui_port):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise SystemExit(f"Port {port} is busy. Choose a different port.") from exc
    environment = {
        **os.environ,
        "LAB_API_URL": f"http://127.0.0.1:{args.api_port}",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }
    commands = [
        [
            sys.executable,
            "-m",
            "uvicorn",
            "knowledge_lab.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.api_port),
            "--workers",
            "1",
        ],
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui.py",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(args.ui_port),
            "--server.headless",
            "true",
        ],
    ]
    children = []

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for command in commands:
            children.append(subprocess.Popen(command, cwd=ROOT, env=environment))
        print(f"Knowledge Lab: http://127.0.0.1:{args.ui_port}")
        print(f"API reference: http://127.0.0.1:{args.api_port}/docs")
        print(
            "No model services are started and no downloads are performed. Ctrl-C stops both apps."
        )
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
