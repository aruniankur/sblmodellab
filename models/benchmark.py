"""Sweep dimensions and layer counts using isolated, real training steps."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = Path(__file__).resolve().parent / "benchmark_results"
FIELDS = ["dim", "layers", "heads", "model_type", "status", "completed_steps", "parameters",
          "n_genes", "micro_batches_per_unit", "seconds_per_unit", "seconds_per_backward",
          "peak_allocated_gb", "peak_reserved_gb", "total_vram_gb", "available_vram_gb",
          "loss", "grad_total", "data_seconds", "wall_seconds", "device", "file", "reason"]


def inclusive_range(start, end, step):
    if start < 1 or end < start or step < 1:
        raise ValueError("Ranges require 1 <= start <= end and a positive step")
    # End is a bound, included when it falls on the requested step grid.
    return list(range(start, end + 1, step))


def isolated_trial(cfg, options, timeout):
    """A timed-out/OOM child cannot retain memory for the next candidate."""
    base = {"dim": cfg.dim, "layers": cfg.layers, "heads": cfg.heads,
            "model_type": cfg.model_type, "file": cfg.file}
    try:
        cfg.validate()
    except ValueError as exc:
        return {**base, "status": "invalid", "reason": str(exc)}
    with tempfile.TemporaryDirectory(prefix="tahoe_capacity_") as temporary:
        request = Path(temporary) / "request.json"
        result = Path(temporary) / "result.json"
        request.write_text(json.dumps({"config": cfg.to_dict(), "trial": options}), encoding="utf-8")
        command = [sys.executable, "-m", "models.training.capacity", "--request", str(request),
                   "--result", str(result)]
        start = time.perf_counter()
        process = subprocess.Popen(command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        try:
            _, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return {**base, "status": "timeout", "wall_seconds": time.perf_counter() - start,
                    "reason": f"Exceeded {timeout:g}s including startup/data preparation; fit is unconfirmed"}
        except BaseException:
            process.kill()
            process.communicate()
            raise
        if process.returncode != 0 or not result.is_file():
            return {**base, "status": "error", "wall_seconds": time.perf_counter() - start,
                    "reason": f"Worker exit {process.returncode}: {stderr[-2000:]}"}
        return {**base, **json.loads(result.read_text(encoding="utf-8")),
                "wall_seconds": time.perf_counter() - start}


def sweep(cfg, dims, layers, *, output, steps=2, warmup=1, timeout=180., max_vram_fraction=.9):
    from tqdm.auto import tqdm
    from models.training.checkpoints import write_json

    if not dims or not layers or steps < 1 or warmup < 0 or timeout <= 0 or not 0 < max_vram_fraction <= 1:
        raise ValueError("Need nonempty ranges, steps >= 1, warmup >= 0, timeout > 0 and VRAM fraction in (0,1]")
    if cfg.model_type == "control":
        # The control architecture ignores layers. Test one effective shape per dim.
        layers = [layers[0]]
        print("Control model has no layer stack: testing one trial per dim; layers have no effect.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    options = dict(steps=steps, warmup=warmup, max_vram_fraction=max_vram_fraction)
    info = {"created": datetime.now().isoformat(timespec="seconds"), "status": "running",
            "config": cfg.to_dict(), "dims": list(dims), "layers": list(layers),
            "trial": options, "timeout_seconds": timeout, "results": []}
    write_json(output / "summary.json", info)
    rows = info["results"]
    try:
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            fh.flush()
            shapes = [(dim, depth) for dim in dims for depth in layers]
            for dim, depth in tqdm(shapes, desc="Training capacity", unit="shape"):
                row = isolated_trial(replace(cfg, dim=dim, layers=depth), options, timeout)
                rows.append(row)
                writer.writerow(row)
                fh.flush()
                write_json(output / "summary.json", info)
                seconds = row.get("seconds_per_unit")
                timing = f"{seconds:.2f}s/unit" if seconds is not None else row.get("reason", "")
                tqdm.write(f"dim={dim}, layers={depth}: {row['status']} | "
                           f"peak reserved {row.get('peak_reserved_gb', 0):.2f} GB | {timing}")
        info["status"] = "finished"
    except KeyboardInterrupt:
        info["status"] = "interrupted"
        raise
    except Exception as exc:
        info.update(status="failed", error=str(exc))
        raise
    finally:
        fits = [r for r in rows if r["status"] == "fits"]
        # Report the frontier, not one ambiguous scalar called the maximum model.
        info["largest_fitting_dim_by_layers"] = {
            str(depth): max((r["dim"] for r in fits if r["layers"] == depth), default=None)
            for depth in layers}
        write_json(output / "summary.json", info)
    print(f"\nResults: {output}\nLargest fitting dim per layer count: {info['largest_fitting_dim_by_layers']}")
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("full", "control"), default="full")
    parser.add_argument("--file", help="Real plate1..12 H5AD; otherwise choose interactively")
    parser.add_argument("--root", default=None)
    for flag in ("dim-start", "dim-end", "dim-step", "layers-start", "layers-end"):
        parser.add_argument(f"--{flag}", type=int, default=None)
    parser.add_argument("--layers-step", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--conc-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=2, help="Timed optimizer units per shape")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=180., help="Hard seconds per isolated shape, including startup")
    parser.add_argument("--max-vram-fraction", type=float, default=.9,
                        help="Maximum fraction of initially available VRAM for a fits verdict")
    parser.add_argument("--amp", choices=("off", "bf16"), default=None)
    parser.add_argument("--query-chunk", type=int, default=4096)
    parser.add_argument("--readout-chunk", type=int, default=16)
    parser.add_argument("--n-control", type=int, default=256)
    parser.add_argument("--n-target", type=int, default=128)
    parser.add_argument("--min-cells", type=int, default=128)
    parser.add_argument("--batches-per-unit", type=int, default=2)
    parser.add_argument("--atom-cache", default=None)
    parser.add_argument("--out", default=None, help="New report folder; existing folders are never overwritten")
    return parser, parser.parse_args(argv)


def main(argv=None):
    parser, args = parse_args(argv)
    # Reuse library/GPU checks; do not import the training stack before these checks.
    from models.run import check_environment, number
    detected_amp = check_environment()
    from models.cli import choose_file
    from models.training.config import TrainingConfig
    from models.training.data import discover_files

    for key, label, default in (("dim_start", "dim start", 256), ("dim_end", "dim end", 1024),
                                 ("dim_step", "dim step", 256), ("layers_start", "layers start", 1),
                                 ("layers_end", "layers end", 4)):
        if getattr(args, key) is None:
            setattr(args, key, number(label, default))
    try:
        dims = inclusive_range(args.dim_start, args.dim_end, args.dim_step)
        layers = inclusive_range(args.layers_start, args.layers_end, args.layers_step)
    except ValueError as exc:
        parser.error(str(exc))
    kwargs = {key: getattr(args, key) for key in ("heads", "conc_dim", "query_chunk", "readout_chunk",
                                                "n_control", "n_target", "min_cells", "batches_per_unit")}
    cfg = TrainingConfig(model_type=args.model, mode="single", amp=args.amp or detected_amp, **kwargs)
    if args.root:
        cfg.root = str(Path(args.root).expanduser().resolve())
    cfg.file = choose_file(cfg.root, args.file)
    cfg.file = discover_files(cfg)[0]["path"]
    if args.atom_cache:
        cfg.atom_cache = str(Path(args.atom_cache).expanduser().resolve())
    if cfg.model_type == "full" and not Path(cfg.atom_cache).is_file():
        parser.error("Chemical cache is missing; run python models/molformer_atoms.py first")
    if args.steps < 1 or args.warmup < 0 or args.timeout <= 0 or not 0 < args.max_vram_fraction <= 1:
        parser.error("steps >= 1, warmup >= 0, timeout > 0 and max-vram-fraction in (0,1] are required")
    output = Path(args.out) if args.out else DEFAULT_RESULTS / f"benchmark_{datetime.now():%Y%m%d_%H%M%S_%f}"
    print(f"Testing {len(dims)} dims x {len(layers)} layer counts, heads={cfg.heads}, "
          f"conc_dim={cfg.conc_dim}; chunks={cfg.query_chunk}/{cfg.readout_chunk}; amp={cfg.amp}")
    print("Every candidate performs training updates; benchmark weights are discarded.")
    sweep(cfg, dims, layers, output=output, steps=args.steps, warmup=args.warmup,
          timeout=args.timeout, max_vram_fraction=args.max_vram_fraction)


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        raise SystemExit("Interactive input ended; benchmark was not started.")
    except KeyboardInterrupt:
        raise SystemExit("Benchmark interrupted; completed results are saved.")
