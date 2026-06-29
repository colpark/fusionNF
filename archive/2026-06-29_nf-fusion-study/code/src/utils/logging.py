"""Lightweight run logging: append-only JSONL + a CSV mirror, no heavy deps.

A single JsonlLogger per run records arbitrary metric dicts per step. We keep both
JSONL (schema-flexible) and a flat CSV (easy to plot/diff) because the brief asks
for "a single CSV/JSONL per run".
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        self.csv_path = os.path.join(run_dir, "metrics.csv")
        self._fields: list[str] = []
        self._rows: list[dict] = []
        # truncate any prior partial logs for a clean, reproducible run dir
        open(self.jsonl_path, "w").close()

    def log(self, **record: Any) -> None:
        with open(self.jsonl_path, "a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        self._rows.append(record)
        for k in record:
            if k not in self._fields:
                self._fields.append(k)
        self._flush_csv()

    def _flush_csv(self) -> None:
        with open(self.csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self._fields)
            w.writeheader()
            for r in self._rows:
                w.writerow(r)

    def summary(self, **record: Any) -> None:
        with open(os.path.join(self.run_dir, "summary.json"), "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
