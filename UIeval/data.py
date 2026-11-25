"""Incremental, read-only readers for logs that are still being written."""
from __future__ import annotations

import csv
import io
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[1] / "models" / "trained_model"
TEXT_COLUMNS = {"file", "cell_line", "control_sample", "target_sample", "drug", "unit"}


def discover_runs(root):
    root = Path(root).expanduser()
    try:
        if not root.is_dir():
            return []
        children = list(root.iterdir())
    except OSError:
        return []
    return sorted((p for p in children if p.is_dir() and not p.is_symlink()
                   and any((p / name).is_file() for name in ("log.csv", "config.json", "info.json"))),
                  key=lambda p: p.name, reverse=True)


@dataclass
class LogSnapshot:
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)
    modified: float | None = None
    pending: bool = False


class LiveLogReader:
    """Read only appended complete records; reset after resume truncation/replacement.

    The end-of-prefix anchor detects rewritten files that have already grown past
    the previous byte offset. A new file handle is used on every poll so atomic
    checkpoint-resume replacement works on Windows as well as Linux.
    """
    def __init__(self):
        self.identity = None
        self.offset = 0
        self.anchor = b""
        self.header = None
        self.frame = pd.DataFrame()
        self.modified_ns = None
        self.invalid_rows = 0

    def reset(self):
        self.identity = None
        self.offset = 0
        self.anchor = b""
        self.header = None
        self.frame = pd.DataFrame()
        self.invalid_rows = 0

    def poll(self, path):
        notes = []
        try:
            with Path(path).open("rb") as fh:
                stat = os.fstat(fh.fileno())
                identity = (stat.st_dev, stat.st_ino)
                changed = (self.identity is not None and identity != self.identity)
                changed |= stat.st_size < self.offset
                if self.offset and self.anchor:
                    fh.seek(self.offset - len(self.anchor))
                    changed |= fh.read(len(self.anchor)) != self.anchor
                if (self.modified_ns is not None and stat.st_mtime_ns != self.modified_ns
                        and stat.st_size == self.offset):
                    changed = True
                if changed:
                    self.reset()
                    notes.append("The log was replaced or rewound; the view has been reloaded.")
                self.identity = identity
                self.modified_ns = stat.st_mtime_ns
                fh.seek(self.offset)
                appended = fh.read()
                end = appended.rfind(b"\n") + 1
                complete = appended[:end]
                pending = end < len(appended)
                if complete:
                    try:
                        records = list(csv.reader(io.StringIO(complete.decode("utf-8-sig"), newline=""), strict=True))
                    except (csv.Error, UnicodeDecodeError):
                        # A newline can occur inside a partially written quoted record.
                        return LogSnapshot(self.frame, notes + ["Waiting for a complete CSV record."], stat.st_mtime, True)
                    header = self.header
                    if header is None and records:
                        header = records.pop(0)
                        if not header or len(set(header)) != len(header) or "loss" not in header:
                            return LogSnapshot(pd.DataFrame(), ["The log header must contain a unique loss column."], stat.st_mtime)
                    valid = []
                    for row in records:
                        if len(row) != len(header) or row == header:
                            self.invalid_rows += 1
                            continue
                        valid.append(row)
                    if valid:
                        new = pd.DataFrame(valid, columns=header)
                        for column in new.columns:
                            if column not in TEXT_COLUMNS:
                                new[column] = pd.to_numeric(new[column], errors="coerce")
                        required = [c for c in ("loss", "epoch", "backward_pass", "optimizer_step") if c in new]
                        good = pd.Series(True, index=new.index)
                        for column in required:
                            good &= new[column].map(lambda value: pd.notna(value) and math.isfinite(float(value)))
                        self.invalid_rows += int((~good).sum())
                        new = new.loc[good]
                        self.frame = pd.concat([self.frame, new], ignore_index=True) if not self.frame.empty else new.reset_index(drop=True)
                    self.header = header
                    self.offset += len(complete)
                    fh.seek(max(0, self.offset - 512))
                    self.anchor = fh.read(min(512, self.offset))
                if pending:
                    notes.append("The last row is still being written; it will appear when complete.")
                if self.invalid_rows:
                    notes.append(f"Skipped {self.invalid_rows:,} malformed or non-finite records.")
                return LogSnapshot(self.frame, notes, stat.st_mtime, pending)
        except FileNotFoundError:
            self.reset()
            self.modified_ns = None
            return LogSnapshot(warnings=["Waiting for log.csv to be created."])
        except OSError as exc:
            return LogSnapshot(self.frame, [f"Log temporarily unavailable: {exc}"], pending=True)


class RunReader:
    def __init__(self):
        self.log = LiveLogReader()
        self.json_cache = {}

    def read_json(self, path):
        path = Path(path)
        previous = self.json_cache.get(path)
        try:
            stat = path.stat()
            stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if previous and previous[0] == stamp:
                return previous[1], None
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            self.json_cache[path] = (stamp, value)
            return value, None
        except (OSError, ValueError) as exc:
            # Epoch metadata is atomically replaced by training; retain the last good view.
            return (previous[1] if previous else {}), f"{path.name} temporarily unavailable: {exc}"

    def poll(self, run):
        run = Path(run)
        log = self.log.poll(run / "log.csv")
        info, info_note = self.read_json(run / "info.json")
        config, config_note = self.read_json(run / "config.json")
        if not config:
            config = info.get("config", {})
        return log, info, config, [note for note in (info_note, config_note) if note]


def filter_frame(frame, *, epochs=(), plates=(), lines=()):
    filtered = frame
    for column, selection in (("epoch", epochs), ("plate", plates), ("cell_line", lines)):
        if selection and column in filtered:
            filtered = filtered.loc[filtered[column].isin(selection)]
    return filtered


def epoch_scores(info):
    records = info.get("epochs", [])
    if not isinstance(records, list):
        return pd.DataFrame()
    frame = pd.DataFrame([r for r in records if isinstance(r, dict)])
    if "epoch" not in frame:
        return pd.DataFrame()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["epoch"]).sort_values("epoch")


def x_values(frame, axis="Backward pass"):
    column = "optimizer_step" if axis == "Optimizer step" else "backward_pass"
    if column in frame:
        return frame[column], axis
    if "step" in frame:
        return frame["step"], "Step"
    return pd.Series(frame.index + 1, index=frame.index), "Log row"


def smoothing(series, window, median=False):
    series = pd.to_numeric(series, errors="coerce").replace([float("inf"), -float("inf")], float("nan"))
    rolling = series.rolling(window=max(1, int(window)), min_periods=1)
    return rolling.median() if median else rolling.mean()


def downsample(frame, limit=1800):
    if len(frame) <= limit:
        return frame
    stride = math.ceil(len(frame) / limit)
    selected = frame.iloc[::stride]
    if selected.index[-1] != frame.index[-1]:
        selected = pd.concat([selected, frame.tail(1)])
    return selected
