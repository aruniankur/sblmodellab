"""Select and ZIP a complete trained-model run using only the standard library."""
from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

MODELS_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNS = MODELS_ROOT / "trained_model"
DEFAULT_OUTPUT = MODELS_ROOT / "trained_model_zips"


def available_runs(root):
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and not p.is_symlink()),
                  key=lambda p: p.name, reverse=True)


def choose_run(root, name=None):
    runs = available_runs(root)
    if not runs:
        raise ValueError(f"No run folders found in {Path(root).resolve()}")
    if name:
        matches = [p for p in runs if p.name == name]
        if not matches:
            raise ValueError(f"No run named {name!r} in {root}")
        return matches[0]
    print(f"\nTrained models in {Path(root).resolve()}:\n")
    for index, run in enumerate(runs, 1):
        weights = [p.name for p in run.glob('*.pt') if p.is_file()]
        print(f"  {index}. {run.name}  [{', '.join(weights) or 'no .pt files'}]")
    while True:
        answer = input("\nFolder number or exact name (q to cancel): ").strip()
        if answer.lower() in ("q", "quit", "exit"):
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(runs):
            return runs[int(answer) - 1]
        for run in runs:
            if answer == run.name:
                return run
        print("Select one of the listed folders.")


def snapshot(run):
    """Capture finalized files and empty directories, detecting concurrent writes."""
    records = {}
    for path in sorted(run.rglob('*')):
        if path.is_symlink():
            raise ValueError(f"Symbolic links are not supported in a model export: {path}")
        if path.name.endswith(('.tmp', '.partial')):
            continue
        relative = path.relative_to(run)
        if path.is_dir():
            records[str(relative)] = ('directory', 0, 0)
        elif path.is_file():
            stat = path.stat()
            records[str(relative)] = ('file', stat.st_size, stat.st_mtime_ns)
    return records


def archive_run(run, output=DEFAULT_OUTPUT):
    run = Path(run).resolve()
    output = Path(output).resolve()
    if not run.is_dir():
        raise ValueError(f"Run folder does not exist: {run}")
    if output == run or run in output.parents:
        raise ValueError("ZIP output must be outside the selected run folder")
    before = snapshot(run)
    files = [name for name, record in before.items() if record[0] == 'file']
    if not files:
        raise ValueError(f"Selected folder contains no finalized files: {run}")
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination = output / f'{run.name}_export_{stamp}.zip'
    if destination.exists():
        raise FileExistsError(destination)
    handle, temporary = tempfile.mkstemp(prefix=destination.stem + '_', suffix='.partial', dir=output)
    os.close(handle)
    temporary = Path(temporary)
    try:
        # Compression level 1 prioritizes server export speed. ZIP64 supports large weights.
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1, allowZip64=True) as archive:
            archive.writestr(run.name + '/', '')
            completed = 0
            for name, record in before.items():
                path = run / name
                member = (Path(run.name) / name).as_posix()
                if record[0] == 'directory':
                    archive.writestr(member + '/', '')
                else:
                    archive.write(path, member)
                    completed += 1
                    print(f'\rZipping files: {completed}/{len(files)}', end='', flush=True)
        print()
        if snapshot(run) != before:
            raise RuntimeError('Run files changed during export. Retry after training or checkpoint writing finishes.')
        with zipfile.ZipFile(temporary) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f'ZIP integrity check failed: {bad}')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'\nZIP ready: {destination}')
    print(f'Size: {destination.stat().st_size / (1024 ** 2):.1f} MiB; integrity check passed.')
    print('Includes finalized weights, JSON metadata, CSV logs and other run files. Original folder preserved.')
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_RUNS, help='Directory containing trained-model runs')
    parser.add_argument('--folder', help='Exact run folder name; omit to choose interactively')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT, help='Directory for exported ZIP files')
    args = parser.parse_args(argv)
    try:
        selected = choose_run(args.root, args.folder)
        if selected is None:
            print('Cancelled.')
            return None
        return archive_run(selected, args.out)
    except (ValueError, RuntimeError, OSError, zipfile.BadZipFile) as exc:
        raise SystemExit(str(exc))


if __name__ == '__main__':
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit('\nCancelled.')
