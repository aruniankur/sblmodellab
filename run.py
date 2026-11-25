"""Launch the organized training workflow implemented in models/run.py."""
from models.run import main

if __name__ == "__main__":
    try:
        main()
    except EOFError:
        raise SystemExit("Interactive input ended; no further training was started.")
    except KeyboardInterrupt:
        raise SystemExit("Cancelled.")
