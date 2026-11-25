"""File selection helpers for the launcher and training configuration."""

from __future__ import annotations

import os


def ask(prompt: str, default, cast=int):
    """Prompt with a default; an empty Enter or closed stdin keeps the default."""
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    if not raw:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        print(f"  not a valid value, keeping {default}")
        return default


def choose_file(root: str, value: str | None = None, preview: int = 25) -> str:
    """Resolve a file under ``root`` from a path, a name, or a numbered choice."""
    if not os.path.isdir(root):
        raise SystemExit(f"data folder not found: {root}")
    names = sorted(
        n for n in os.listdir(root) if n.startswith("plate") and n.endswith(".h5ad")
    )
    if not names:
        raise SystemExit(f"no plate*.h5ad in {root}")

    if value:
        for candidate in (value, os.path.join(root, value)):
            if os.path.exists(candidate):
                return candidate
        matches = [n for n in names if value.lower() in n.lower()]
        if len(matches) == 1:
            return os.path.join(root, matches[0])
        if not matches:
            raise SystemExit(f"{value!r} matches no file in {root}")
        print(f"{value!r} is ambiguous, {len(matches)} matches:")
        for name in matches[:12]:
            print(f"    {name}")
        value = None

    shown = names[:preview]
    print(f"\n{len(names)} files in {root}" + (
        f" (showing the first {len(shown)})" if len(names) > len(shown) else ""
    ))
    for i, name in enumerate(shown, 1):
        print(f"  {i:3d}. {name}")
    choice = ask("\nfile (number, or part of a name)", "1", str)

    if choice.isdigit():
        index = int(choice)
        if not 1 <= index <= len(shown):
            raise SystemExit(
                f"pick 1..{len(shown)}, or type part of a file name "
                f"(e.g. CVCL_1125) instead of a number"
            )
        return os.path.join(root, shown[index - 1])
    return choose_file(root, choice, preview)
