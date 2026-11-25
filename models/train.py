"""Non-interactive full-grid training; use run.py for the interactive launcher."""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.training.config import parse_args
from models.training.engine import dry_run, train


def main(argv=None):
    args, cfg = parse_args(argv)
    if args.resume:
        return train(resume=args.resume, additional_epochs=args.epochs if args.epochs is not None else 10)
    if args.dry_run:
        return dry_run(cfg)
    return train(cfg)


if __name__ == "__main__":
    main()
