"""Start the API and interface; Ctrl-C shuts down child processes cleanly."""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import httpx

    from summarizer.config import Settings

    settings = Settings()
    children = []
    log_handles = []
    try:
        address = urlsplit(settings.api_base_url)
        if address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost"}:
            raise RuntimeError("The local launcher requires a loopback HTTP API_BASE_URL.")
        port = address.port or 80
        # Fail before starting children if a different application owns the API port.
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is in use. Choose another API_BASE_URL port.") from exc
        # Start only a project-local runtime, and only if the configured local server is down.
        local_binary = ROOT / ".runtime/ollama/ollama"
        if not local_binary.exists():
            local_binary = ROOT / ".runtime/ollama/bin/ollama"
        try:
            httpx.get(settings.ollama_base_url + "/api/version", timeout=2, trust_env=False)
            reachable = True
        except httpx.HTTPError:
            reachable = False
        if (
            not reachable
            and local_binary.exists()
            and settings.ollama_base_url in {"http://127.0.0.1:11434", "http://localhost:11434"}
        ):
            env = os.environ | {
                "OLLAMA_MODELS": str(ROOT / ".runtime/models"),
                "OLLAMA_HOST": "127.0.0.1:11434",
                "OLLAMA_NO_CLOUD": "1",
            }
            handle = (ROOT / ".runtime/ollama.log").open("a")
            log_handles.append(handle)
            children.append(
                subprocess.Popen(
                    [str(local_binary), "serve"], env=env, stdout=handle, stderr=handle
                )
            )
        python = sys.executable
        children.append(
            subprocess.Popen(
                [
                    python,
                    "-m",
                    "uvicorn",
                    "summarizer.api:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ]
            )
        )
        for _ in range(60):
            if children[-1].poll() is not None:
                raise RuntimeError(
                    f"API failed to start. Check whether port {port} is already in use."
                )
            try:
                response = httpx.get(
                    settings.api_base_url + "/openapi.json", timeout=1, trust_env=False
                )
                if (
                    response.is_success
                    and response.json().get("info", {}).get("title")
                    == "Support Transcript Summarizer"
                ):
                    break
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.5)
        else:
            raise RuntimeError("API startup timed out.")
        children.append(
            subprocess.Popen([python, "-m", "streamlit", "run", "ui.py", "--server.port", "8501"])
        )
        print(
            f"\nHandover: http://127.0.0.1:8501\nAPI docs: {settings.api_base_url}/docs\n",
            flush=True,
        )
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        return 1
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for handle in log_handles:
            handle.close()


def stop(*_args):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(main())
