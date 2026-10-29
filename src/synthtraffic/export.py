"""Export: strips label columns and writes the dataset to disk. Draws no random numbers."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .generate import Result

__all__ = ["export_frames", "write_outputs"]


def export_frames(res: Result) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Public tables: sessions, events (no ``_`` columns) and the ground-truth answer key."""
    S = res.sessions
    E = res.events.drop(columns=[c for c in res.events.columns if c.startswith("_")])
    G = (res.gt.rename(columns={"_is_bot": "is_bot", "_anomaly": "anomaly"})
         [["session_id", "date", "is_bot", "anomaly"]])
    if res.cfg.sparse_ground_truth:
        G = G[(G.is_bot == 1) | G.anomaly.notna()]
    return S, E, G


def write_outputs(res: Result, out_dir: str | Path | None = None, *, fmt: str = "csv",
                  report_text: str | None = None, sample_events: int = 3000) -> list[Path]:
    """Write all files and return their paths. ``fmt`` is ``"csv"`` or ``"parquet"``."""
    if fmt not in ("csv", "parquet"):
        raise ValueError(f"fmt must be 'csv' or 'parquet', not {fmt!r}")
    out = Path(out_dir if out_dir is not None else res.cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    S, E, G = export_frames(res)
    tables = {"sessions": S, "daily_kpis": res.kpi, "_ground_truth": G}
    if len(E):
        tables["events"] = E
    paths: list[Path] = []
    for name, df in tables.items():
        p = out / f"{name}.{fmt}"
        if fmt == "csv":
            df.to_csv(p, index=False)
        else:
            df.to_parquet(p, index=False)
        paths.append(p)
    if len(E):
        p = out / "events_sample.jsonl"
        E.head(sample_events).to_json(p, orient="records", lines=True, date_format="iso")
        paths.append(p)
    if res.log:
        p = out / "access.log"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(res.log) + "\n")
        paths.append(p)
    if report_text:
        p = out / "test_report.txt"
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(report_text)
        paths.append(p)
    meta = dict(res.meta, generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                files=[q.name for q in paths])
    p = out / "run_metadata.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    paths.append(p)
    return paths
