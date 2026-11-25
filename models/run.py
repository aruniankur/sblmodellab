"""Interactive launcher. Dependency and GPU checks run before training imports."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check_environment():
    required = {"torch": "torch", "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
                "anndata": "anndata", "h5py": "h5py", "tqdm": "tqdm"}
    problems = []
    print("Checking training libraries:")
    for module, package in required.items():
        try:
            importlib.import_module(module)
            print(f"  {package}: OK")
        except Exception as exc:
            problems.append(package)
            print(f"  {package}: unavailable ({exc})")
    if problems:
        requirements = Path(__file__).resolve().parents[1] / "requirements-training.txt"
        raise SystemExit(f"Install the required libraries with:\n  {sys.executable} -m pip install -r {requirements}")
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu.name}; {gpu.total_memory / 1e9:.2f} GB; CUDA {torch.version.cuda}")
        amp = "bf16" if torch.cuda.is_bf16_supported() else "off"
        print(f"Training precision: {amp}")
        return amp
    print("No CUDA GPU detected. Training will use CPU and FP32 (full-size runs will be slow).")
    return "off"


def choice(prompt, options):
    print(f"\n{prompt}")
    for index, label in enumerate(options, 1):
        print(f"  {index}. {label}")
    while True:
        raw = input("Select: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print(f"Enter a number from 1 to {len(options)}.")


def number(prompt, default, minimum=1):
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        try:
            value = int(raw) if raw else default
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"Enter an integer >= {minimum}.")


def existing_runs(root):
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir()
                   and (p / "last.pt").is_file() and (p / "config.json").is_file()),
                  key=lambda p: p.name, reverse=True)


def main():
    amp = check_environment()
    from models.cli import choose_file
    from models.training.config import TRAINED_ROOT, TrainingConfig
    from models.training.engine import train

    action = choice("Training", ["Train a new model", "Continue an existing model"])
    if action == 1:
        runs = existing_runs(TRAINED_ROOT)
        if not runs:
            print(f"No resumable runs found in {TRAINED_ROOT}")
            return
        descriptions = []
        for run in runs:
            cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
            descriptions.append(f"{run.name} ({cfg['model_type']}, {cfg['mode']})")
        selected = runs[choice("Select a saved run", descriptions)]
        epochs = number("How many more epochs?", 10)
        train(resume=str(selected), additional_epochs=epochs)
        return

    model_type = ("full", "control")[choice("Model", ["Full chemical perturbation model", "Control: DMSO only"])]
    mode = ("all", "single")[choice("Data", ["All eligible cell lines, plates 1-12", "One selected file only"])]
    cfg = TrainingConfig(model_type=model_type, mode=mode, amp=amp)
    if mode == "single":
        cfg.file = choose_file(cfg.root)
        cfg.lr, cfg.dropout, cfg.grad_clip = 3e-4, 0.0, 10.0
    if model_type == "control":
        print("Control has no attention stack: heads/layers are recorded but do not affect computation.")
    while True:
        # These are the only numerical training/model settings requested interactively.
        cfg.dim = number("dim", 512, 4)
        cfg.conc_dim = number("conc_dim", 32)
        cfg.heads = number("heads", 8)
        cfg.layers = number("layers", 1, 0 if model_type == "control" else 1)
        cfg.epochs = number("epochs", 60 if mode == "all" else 10)
        try:
            cfg.validate()
            break
        except ValueError as exc:
            print(f"Invalid configuration: {exc}")
    if model_type == "full" and not Path(cfg.atom_cache).is_file():
        raise SystemExit("Chemical cache is missing. Build it first with: python models/molformer_atoms.py")
    train(cfg)


if __name__ == "__main__":
    try:
        main()
    except EOFError:
        raise SystemExit("Interactive input ended; no further training was started.")
    except KeyboardInterrupt:
        raise SystemExit("Cancelled.")
