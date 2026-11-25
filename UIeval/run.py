"""Start the read-only dashboard and print local/network browser addresses."""
from __future__ import annotations

import argparse
import importlib
import os
import socket
import subprocess
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0", help="Use 127.0.0.1 for SSH forwarding only")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--runs-root", type=Path, help="Override models/trained_model")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    ui_root = Path(__file__).resolve().parent
    missing = []
    for name in ("streamlit", "pandas", "plotly"):
        try:
            module = importlib.import_module(name)
            if name == "streamlit" and (not hasattr(module, "fragment") or
                    tuple(int(v) for v in module.__version__.split(".")[:2]) < (1, 50)):
                missing.append("streamlit>=1.50")
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(f"Missing dashboard libraries: {', '.join(missing)}\nInstall: {sys.executable} -m pip install -r {ui_root / 'requirements.txt'}")
    environment = os.environ.copy()
    if args.runs_root:
        environment["UIEVAL_RUNS_ROOT"] = str(args.runs_root.expanduser().resolve())
    print(f"\nUIeval · Training monitor\nLocal browser: http://localhost:{args.port}", flush=True)
    if args.host == "0.0.0.0":
        try:
            address = socket.gethostbyname(socket.gethostname())
            print(f"Network browser: http://{address}:{args.port} (use a server address reachable from your PC)", flush=True)
        except OSError:
            print(f"Network browser: http://SERVER_IP:{args.port}", flush=True)
    elif args.host not in ("localhost", "127.0.0.1"):
        print(f"Network browser: http://{args.host}:{args.port}", flush=True)
    print("Training runs independently. Stop this dashboard with Ctrl+C.\n", flush=True)
    command = [sys.executable, "-m", "streamlit", "run", str(ui_root / "app.py"),
               f"--server.address={args.host}", f"--server.port={args.port}",
               "--server.headless=true", "--browser.gatherUsageStats=false"]
    try:
        return subprocess.call(command, cwd=ui_root, env=environment)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
