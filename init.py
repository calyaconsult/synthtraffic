"""synthtraffic - synthetic web-traffic data for a travel / tour-operator website.

Quick start::

    from synthtraffic import Config, run, run_tests, write_outputs
    res = run(Config(seed=42))
    results, metrics = run_tests(res)
    write_outputs(res, "outputs")

Importing the package has no side effects: no I/O and no random draws.
"""
from __future__ import annotations

from .config import Config, load
from .export import export_frames, write_outputs
from .generate import Result, run
from .model import EVENT_CAL, SITE_TREE, TOUR2COUNTRY, Site, build_site
from .validate import TARGETS, anomaly_dates, kpi_metrics, report, run_tests

__version__ = "2.0.0"

__all__ = [
    "__version__", "Config", "load", "Site", "build_site", "SITE_TREE", "TOUR2COUNTRY",
    "EVENT_CAL", "Result", "run", "export_frames", "write_outputs", "TARGETS",
    "anomaly_dates", "kpi_metrics", "run_tests", "report",
]
