"""Command-line interface: ``synthtraffic ...`` or ``python -m synthtraffic ...``.

Exit codes: 0 = success, 1 = at least one check failed, 2 = invalid arguments or config.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from typing import Sequence

from . import __version__
from .config import Config, load
from .export import write_outputs
from .generate import run
from .validate import report, run_tests

log = logging.getLogger("synthtraffic")
FULL_WINDOW = 91          # the KPI bands are calibrated on a 91-day window


class _StderrHandler(logging.Handler):
    """Writes to the *current* sys.stderr (works with pytest's capsys)."""

    def emit(self, record: logging.LogRecord) -> None:
        print(self.format(record), file=sys.stderr)


def _setup_logging(verbose: int) -> None:
    for h in list(log.handlers):
        log.removeHandler(h)
    h = _StderrHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    log.addHandler(h)
    log.propagate = False
    log.setLevel(logging.DEBUG if verbose > 1 else logging.INFO if verbose else logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="synthtraffic",
        description="Generate a synthetic web-traffic dataset for a travel website.",
        epilog="Exit codes: 0 ok, 1 checks failed, 2 invalid arguments or config.")
    g = p.add_argument_group("configuration (command-line values override the YAML file)")
    g.add_argument("--config", metavar="YAML", help="YAML file with Config fields")
    g.add_argument("--seed", type=int)
    g.add_argument("--start-date", metavar="YYYY-MM-DD")
    g.add_argument("--n-days", type=int, metavar="N")
    g.add_argument("--access-log-lines", type=int, metavar="N")
    g.add_argument("--no-events", action="store_true",
                   help="skip events.csv, the JSONL sample and access.log")
    o = p.add_argument_group("output")
    o.add_argument("--out-dir", metavar="DIR")
    o.add_argument("--format", choices=("csv", "parquet"), default="csv")
    o.add_argument("--dry-run", action="store_true",
                   help="generate and validate, but write nothing")
    v = p.add_argument_group("validation")
    v.add_argument("--validate", action="store_true",
                   help="run the checks and write test_report.txt; exit 1 if any fails")
    v.add_argument("--integrity-only", action="store_true",
                   help="skip the statistical KPI bands (automatic below 91 days)")
    v.add_argument("--seeds", type=int, default=1, metavar="N",
                   help="validate N consecutive seeds starting at --seed; writes nothing")
    p.add_argument("-v", "--verbose", action="count", default=0)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _multi_seed(cfg: Config, n: int, kpi: bool, verbose: int) -> int:
    n_fail = 0
    for s in range(cfg.seed, cfg.seed + n):
        T, M = run_tests(run(replace(cfg, seed=s)), kpi_checks=kpi)
        ok = bool(T.passed.all())
        text = report(T, M, s)
        if verbose:
            print(text)
        else:
            print(text.splitlines()[0] + ("" if ok else "  FAILED: " + ", ".join(T.test[~T.passed])))
        n_fail += not ok
    print(f"{n - n_fail}/{n} seeds passed all checks")
    return 1 if n_fail else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    a = parser.parse_args(argv)
    _setup_logging(a.verbose)
    if a.seeds < 1:
        parser.error("--seeds must be >= 1")
    overrides = dict(seed=a.seed, start_date=a.start_date, n_days=a.n_days,
                     access_log_lines=a.access_log_lines, out_dir=a.out_dir,
                     write_events=False if a.no_events else None)
    try:
        cfg = load(a.config, **overrides)
    except (ValueError, ImportError, OSError) as exc:
        print(f"synthtraffic: error: {exc}", file=sys.stderr)
        return 2

    checking = a.validate or a.dry_run or a.seeds > 1
    kpi = not a.integrity_only
    if checking and kpi and cfg.n_days < FULL_WINDOW:
        log.warning("n_days=%d < %d: KPI bands need the full window, "
                    "running integrity checks only", cfg.n_days, FULL_WINDOW)
        kpi = False
    if a.seeds > 1:
        return _multi_seed(cfg, a.seeds, kpi, a.verbose)

    t0 = time.perf_counter()
    res = run(cfg)
    log.info("generated %s sessions and %s events in %.1fs (seed %d)",
             f"{res.sessions.session_id.nunique():,}", f"{len(res.events):,}",
             time.perf_counter() - t0, cfg.seed)
    failed, text = False, None
    if checking:
        T, M = run_tests(res, kpi_checks=kpi)
        text = report(T, M, cfg.seed)
        failed = not bool(T.passed.all())
        print(text)
    if a.dry_run:
        print(f"dry run: nothing written ({res.sessions.session_id.nunique():,} sessions, "
              f"{len(res.events):,} events)")
        return 1 if failed else 0
    try:
        paths = write_outputs(res, fmt=a.format, report_text=text)
    except ImportError as exc:
        print(f"synthtraffic: error: {exc} (parquet needs `pip install synthtraffic[parquet]`)",
              file=sys.stderr)
        return 2
    print(f"wrote {len(paths)} files to {paths[0].parent}")
    for p in paths:
        log.info("  %-22s %12s bytes", p.name, f"{p.stat().st_size:,}")
    return 1 if failed else 0


if __name__ == "__main__":            # pragma: no cover
    sys.exit(main())
