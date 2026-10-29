"""Validation.

Two kinds of checks:

* integrity checks hold for any seed and window length (tables agree, IDs are
  consistent, no label leakage, correct DST offsets ...);
* statistical KPI bands (``kpi_checks=True``) need roughly the full 91-day window.
  ACF(7) varies between seeds (about 0.30-0.70), so a single seed can miss its band.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config
from .export import export_frames
from .generate import Result
from .model import EVENT_CAL, FUNNEL

__all__ = ["TARGETS", "anomaly_dates", "kpi_metrics", "report", "run_tests"]

# metric: (low, high), fixed before tuning
TARGETS: Mapping[str, tuple[float, float]] = {
    "conversion_rate": (0.015, 0.025), "funnel_completion": (0.40, 0.60),
    "zero_booking_days": (0.0, 0.05), "acf7_detrended": (0.45, 1.0), "resid_cv": (0.06, 0.12),
    "weekend_weekday": (1.05, 1.30), "bounce_acf1": (0.20, 1.0), "bounce_sd": (0.02, 0.06),
    "duration_acf1": (0.20, 1.0), "sessions_per_user_day": (1.10, 1.50)}
_PCT = {"conversion_rate", "funnel_completion", "zero_booking_days", "resid_cv"}
_IMPOSSIBLE = {("desktop", "Samsung Internet"), ("mobile", "Edge"),
               ("tablet", "Edge"), ("tablet", "Samsung Internet")}


def anomaly_dates(cfg: Config, events_cal: Mapping[int, tuple[float, str]] = EVENT_CAL) -> list:
    start = pd.Timestamp(cfg.start_date)
    return [(start + timedelta(days=k)).date() for k in events_cal if 0 <= k < cfg.n_days]


def kpi_metrics(K: pd.DataFrame, anomaly: list) -> dict[str, float]:
    """Daily-KPI statistics; labelled anomaly days are excluded."""
    K = K.copy(); K["date"] = pd.to_datetime(K["date"]); K = K.set_index("date").sort_index()
    m = ~K.index.isin(pd.to_datetime(list(anomaly)))
    s = K.sessions.astype(float).where(m)
    acf7 = (s / s.rolling(7, center=True, min_periods=5).mean()).autocorr(7)
    y = np.log(K.sessions[m].astype(float).values); tt = np.arange(len(K))[m]; dw = K.index[m].dayofweek
    X = np.column_stack([np.ones(m.sum()), tt] + [(dw == j).astype(float) for j in range(1, 7)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); res_ = y - X @ beta
    g = pd.Series(y - beta[0] - beta[1] * tt).groupby(np.asarray(dw))
    wk = K.sessions[m]
    fe = (K.funnel_entry_rate * K.sessions).sum()
    first, last = s.iloc[:28].mean(), s.iloc[-28:].mean()
    return {"conversion_rate": K.bookings.sum() / K.sessions.sum(),
            "funnel_entry": fe / K.sessions.sum(), "funnel_completion": K.bookings.sum() / fe,
            "zero_booking_days": float((K.bookings == 0).mean()),
            "acf7_detrended": float(acf7),
            "resid_cv": float(np.sqrt((res_ ** 2).sum() / (len(y) - X.shape[1]))),
            "dow_between_within": float(g.mean().var() / g.var().mean()),
            "weekend_weekday": float(wk[dw >= 5].mean() / wk[dw < 5].mean()),
            "bounce_acf1": float(K.bounce_rate.where(m).autocorr(1)),
            "bounce_sd": float(K.bounce_rate.where(m).std()),
            "duration_acf1": float(K.avg_duration.where(m).autocorr(1)),
            "sessions_per_user_day": float((K.sessions / K.users).mean()),
            "growth_first_vs_last_4w": float(last / first - 1)}


def _expected_offset(tz: str, ts: pd.Timestamp) -> str | None:
    """UTC offset of a naive server time via zoneinfo; None if the local time is ambiguous."""
    z = ZoneInfo(tz); dt = ts.to_pydatetime()
    a, b = dt.replace(tzinfo=z, fold=0).utcoffset(), dt.replace(tzinfo=z, fold=1).utcoffset()
    if a != b:
        return None
    mins = int(a.total_seconds() // 60)
    return f"{'+' if mins >= 0 else '-'}{abs(mins) // 60:02d}{abs(mins) % 60:02d}"


def run_tests(res: Result, kpi_checks: bool = True) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run all checks. Returns (one row per check, KPI metrics)."""
    S, E, G, K, cfg, site = res.sessions, res.events, res.gt, res.kpi, res.cfg, res.site
    has_ev = len(E) > 0
    Sx, Ex, _ = export_frames(res)
    Sd = S.drop_duplicates("session_id")
    Ed = E.drop_duplicates("event_id") if has_ev else E
    Sb = Sd.merge(G[["session_id", "_is_bot", "_true_country", "_tzshift"]], on="session_id")
    T: list[dict[str, Any]] = []

    def add(name: str, ok: Any, val: str, target: str) -> None:
        T.append({"test": name, "passed": bool(ok), "value": val, "target": target})

    def skip(name: str, why: str) -> None:
        T.append({"test": name, "passed": True, "value": f"skipped ({why})", "target": "-"})

    ev_tests = ("Revenue: events == sessions", "1 purchase event per booking, 0 for others",
                "Confirm page at most once per session", "Pageviews == event rows")
    if has_ev:
        rs, rv = Sd.revenue_eur.sum(), Ed.revenue_eur.sum()
        add(ev_tests[0], abs(rs - rv) < 0.01, f"EUR {rv:,.2f} vs {rs:,.2f}", "equal")
        pc = Ed[Ed.event_name == "purchase"].groupby("session_id").size()
        conv = set(Sd.session_id[Sd.converted == 1])
        add(ev_tests[1], set(pc.index) == conv and (pc == 1).all(),
            f"{len(pc)} purchase sessions / {len(conv)} bookings, max {pc.max() if len(pc) else 0}", "1:1")
        cc = Ed[Ed.page_url == FUNNEL[2]].groupby("session_id").size()
        add(ev_tests[2], (cc <= 1).all(), f"max {cc.max() if len(cc) else 0}", "<= 1")
        ec = Ed.groupby("session_id").size().reindex(Sd.session_id).fillna(0).values
        n_bad = int((ec != Sd.pageviews.values).sum())
        add(ev_tests[3], n_bad == 0, f"{n_bad} mismatches", "0")
    else:
        for t in ev_tests:
            skip(t, "no events")

    o = Sd.sort_values(["user_id", "session_start", "visit_number"])
    add("is_returning == (visit_number > 1)", (o.is_returning == (o.visit_number > 1)).all(),
        f"{int((o.is_returning != (o.visit_number > 1)).sum())} contradictions", "0")
    dv = o.groupby("user_id").visit_number.diff().dropna()
    dt = o.groupby("user_id").session_start.diff().dropna()
    add("visit_number +1 per user, chronological", (dv == 1).all() and (dt >= pd.Timedelta(0)).all(),
        f"{int((dv != 1).sum())} gaps, {int((dt < pd.Timedelta(0)).sum())} out of order", "0 / 0")
    bk = Sb[Sb["_is_bot"] == 1].user_id
    add("Bot IDs unique and never shared with humans",
        bk.is_unique and not bk.isin(set(Sb[Sb["_is_bot"] == 0].user_id)).any(),
        f"{len(bk)} bot sessions, {len(bk) - bk.nunique()} reused IDs", "0 reused")
    peaks = {}
    for c in ("CH", "US", "AE"):
        h = Sb[(Sb["_is_bot"] == 0) & (Sb["_true_country"] == c)]
        peaks[c] = int(((h.hour + h["_tzshift"]) % 24).value_counts().idxmax()) if len(h) else -1
    if cfg.n_days >= 91:
        add("Local-time peak 19-21h (CH, US, AE)", all(19 <= v <= 21 for v in peaks.values()),
            ", ".join(f"{k} {v}:00" for k, v in peaks.items()), "19-21")
    else:
        add("Local-time peak 18-22h (CH, US, AE)", all(18 <= v <= 22 for v in peaks.values()),
            ", ".join(f"{k} {v}:00" for k, v in peaks.items()), "18-22")

    if res.log:
        bad = amb = 0
        for line in res.log:
            mt = re.search(r"\[(\S+) ([+-]\d{4})\]", line)
            if mt is None:
                continue
            ts = pd.to_datetime(mt.group(1), format="%d/%b/%Y:%H:%M:%S")
            exp = _expected_offset(cfg.server_tz, ts)
            if exp is None:
                amb += 1
            else:
                bad += mt.group(2) != exp
        add("access.log DST offsets", bad == 0,
            f"{bad} wrong of {len(res.log)} ({amb} ambiguous skipped)", "0")
    else:
        skip("access.log DST offsets", "no log")

    ds = S.session_id[S.session_id.duplicated()].nunique()
    if has_ev:
        de = E[E.event_id.duplicated()].session_id.nunique()
        add("Duplicates consistent across tables", ds == de and ds > 0,
            f"{ds} dup sessions / {de} in events", "equal, >0")
    else:
        add("Duplicates consistent across tables", ds > 0, f"{ds} dup sessions (no events)", ">0")
    leak = [c for c in list(Sx.columns) + list(Ex.columns) if "anomaly" in c or "bot" in c]
    add("No label columns in sessions/events", not leak, ", ".join(leak) or "none", "none")
    bd = set(zip(Sd.device, Sd.browser)) & _IMPOSSIBLE
    add("Browser consistent with device", not bd, str(bd or "ok"), "no impossible pairs")
    hb = Sb[(Sb["_is_bot"] == 0) & (Sb.pageviews == 1) & (Sb.duration_sec >= 0)]
    add("Bounces: duration mostly 0, max 25s",
        (hb.duration_sec == 0).mean() >= .8 and hb.duration_sec.max() <= 25,
        f"{(hb.duration_sec == 0).mean():.0%} zero, max {hb.duration_sec.max():.0f}s", ">=80%, <=25s")
    md = Sb[(Sb["_is_bot"] == 0) & (Sb.pageviews > 1) & (Sb.duration_sec > 0)]
    rr = np.corrcoef(np.log(md.pageviews), np.log(md.duration_sec))[0, 1]
    add("Duration coupled to depth: corr(log pv, log dur)", rr > .3, f"{rr:.2f}", "> 0.30")
    lo, hi = cfg.price_adj_bounds
    bb = Sd[Sd.converted == 1]
    ratio = bb.revenue_eur / (bb.booked_tour.map(site.price) * bb.pax)
    add("Revenue = price x pax x noise (clean EUR)",
        ratio.between(lo - 1e-3, hi + 1e-3).all(),
        f"{ratio.min():.2f}-{ratio.max():.2f}" if len(ratio) else "no bookings",
        f"{lo:.2f}-{hi:.2f} (exact)")
    ms = (Sd.country == "(not set)").mean()
    add("Missing geo share", .02 <= ms <= .05, f"{ms:.1%}", "2-5%")

    M = kpi_metrics(K, anomaly_dates(cfg, res.events_cal))
    if kpi_checks:
        for k, (lo_, hi_) in TARGETS.items():
            f = (lambda x: f"{x:.2%}") if k in _PCT else (lambda x: f"{x:.2f}")
            add(f"KPI {k}", lo_ <= M[k] <= hi_, f(M[k]),
                f"{f(lo_)}-{f(hi_)}" if hi_ != 1.0 else f">= {f(lo_)}")
    return pd.DataFrame(T), M


def report(T: pd.DataFrame, M: dict[str, float], seed: int) -> str:
    lines = [f"seed {seed}: {int(T.passed.sum())}/{len(T)} passed"]
    lines += [f"  [{'PASS' if r.passed else 'FAIL'}] {r.test:<48} {r.value:<40} target {r.target}"
              for r in T.itertuples()]
    lines += [(f"  info: funnel entry {M['funnel_entry']:.2%}, DOW between/within "
              f"{M['dow_between_within']:.2f}, growth first->last 4 weeks "
              f"{M['growth_first_vs_last_4w']:+.0%}")]
    return "\n".join(lines)
