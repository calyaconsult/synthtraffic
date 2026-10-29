"""Run configuration: a frozen, validated dataclass plus a YAML/override loader."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

__all__ = ["Config", "load"]

_TUPLE_FIELDS = ("funnel_step", "price_adj_bounds")


@dataclass(frozen=True)
class Config:
    seed: int = 42
    start_date: str = "2026-08-01"
    n_days: int = 91
    burn_in_days: int = 28              # simulated history before start_date (not exported)
    base_sessions: int = 520
    growth_per_year: float = 0.55
    nb_dispersion: float = 120.0        # NegBin r: residual CV ~9-11 %
    return_share: float = 0.38          # share of human sessions from known users
    same_day_share: float = 0.45        # of those: revisit on the same day
    return_lag_mean: float = 6.0        # otherwise: lag in days ~ 1 + Exp(mean)
    churn_user_share: float = 0.12      # users whose cookie resets now and then
    device_switch: float = 0.15
    quality_phi: float = 0.75           # AR(1) day-level traffic quality
    quality_sd: float = 0.12
    tour_to_funnel: float = 0.22
    funnel_step: tuple[float, float] = (0.72, 0.80)
    price_adj_bounds: tuple[float, float] = (0.75, 1.35)   # max discount -25 %, upsell +35 %
    bot_share: float = 0.025
    missing_geo_rate: float = 0.035
    dup_session_rate: float = 0.002
    neg_duration_rate: float = 0.004
    sparse_ground_truth: bool = True    # only labelled sessions in _ground_truth.csv
    write_events: bool = True
    access_log_lines: int = 1200
    server_tz: str = "Europe/Zurich"
    out_dir: str = "outputs"

    def __post_init__(self) -> None:
        for f in _TUPLE_FIELDS:                        # YAML gives lists -> keep immutable
            object.__setattr__(self, f, tuple(getattr(self, f)))
        e: list[str] = []

        def prob(name: str, hi_open: bool = False, hi: float = 1.0) -> None:
            v = getattr(self, name)
            if not (0.0 <= v < hi if hi_open else 0.0 <= v <= hi):
                e.append(f"{name}={v} must be in [0, {hi}{')' if hi_open else ']'}")

        if not isinstance(self.seed, int) or self.seed < 0:
            e.append(f"seed={self.seed!r} must be a non-negative int")
        try:
            pd.Timestamp(self.start_date)
        except (ValueError, TypeError):
            e.append(f"start_date={self.start_date!r} is not a date")
        if self.n_days < 14:
            e.append(f"n_days={self.n_days} must be >= 14 (weekly metrics)")
        if self.burn_in_days < 0:
            e.append("burn_in_days must be >= 0")
        if self.base_sessions <= 0:
            e.append("base_sessions must be > 0")
        if self.nb_dispersion <= 0:
            e.append("nb_dispersion must be > 0")
        if self.return_lag_mean <= 0:
            e.append("return_lag_mean must be > 0")
        for n in ("return_share", "same_day_share", "churn_user_share", "device_switch",
                  "tour_to_funnel", "missing_geo_rate", "dup_session_rate", "neg_duration_rate"):
            prob(n)
        prob("bot_share", hi_open=True, hi=0.5)
        if not -1.0 < self.quality_phi < 1.0:
            e.append(f"quality_phi={self.quality_phi} must be in (-1, 1) (stationary AR(1))")
        if self.quality_sd < 0:
            e.append("quality_sd must be >= 0")
        if len(self.funnel_step) != 2 or not all(0 < s <= 1 for s in self.funnel_step):
            e.append(f"funnel_step={self.funnel_step} needs 2 values in (0, 1]")
        lo, hi = (self.price_adj_bounds + (None, None))[:2]
        if len(self.price_adj_bounds) != 2 or not (0 < lo <= 1.0 <= hi):
            e.append(f"price_adj_bounds={self.price_adj_bounds} must satisfy 0 < lo <= 1 <= hi")
        if self.access_log_lines < 0:
            e.append("access_log_lines must be >= 0")
        try:
            ZoneInfo(self.server_tz)
        except (ZoneInfoNotFoundError, ValueError):
            e.append(f"server_tz={self.server_tz!r} is not an IANA time zone")
        if e:
            raise ValueError("invalid config: " + "; ".join(e))

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (tuples become lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def load(path: str | Path | None = None, **overrides: Any) -> Config:
    """Defaults <- YAML file <- keyword overrides (``None`` values are ignored)."""
    data: dict[str, Any] = {}
    if path is not None:
        try:
            import yaml
        except ImportError as exc:                      # optional dependency
            raise ImportError("YAML configs need `pip install synthtraffic[yaml]`") from exc
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: top level must be a mapping")
    data.update({k: v for k, v in overrides.items() if v is not None})
    unknown = sorted(set(data) - {f.name for f in fields(Config)})
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return replace(Config(), **data)
