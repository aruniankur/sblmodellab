"""Convenience launcher for models/zipfolder.py."""
from models.zipfolder import main

if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit('\nCancelled.')
