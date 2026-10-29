"""Generator: turns a Config + Site into sessions, events, ground truth, KPIs and a log.

The order of random draws is identical to the single-file ``generator_v2.py``;
seed 42 therefore reproduces the original dataset exactly.
"""
from __future__ import annotations

import math
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .model import (
    BROWSER_BY_DEVICE,
    CAMPAIGNS,
    CH_CAT,
    CHANNELS,
    COUNTRIES,
    CTY_CAT,
    DEV_CAT,
    DEVICES,
    DOW_W,
    EVENT_CAL,
    FUNNEL,
    FUNNEL_EVENT,
    INTENT_CAT,
    REFERRERS,
    Site,
    build_site,
    hour_cat,
    landing,
    next_page,
    pick,
    season,
    sigmoid,
    utc_offset_h,
)

__all__ = ["Result", "access_log", "daily_kpis", "inject_dirt", "run"]

UA = {"mobile": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
      "desktop": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/127.0 Safari/537.36",
      "tablet": "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
      "bot": "Mozilla/5.0 (compatible; AhrefsBot/7.0; +http://ahrefs.com/robot/)"}


@dataclass
class Result:
    """Everything one run produces. Label columns (``_is_bot`` ...) live only in ``gt``."""
    sessions: pd.DataFrame
    events: pd.DataFrame
    gt: pd.DataFrame
    kpi: pd.DataFrame
    log: list[str]
    cfg: Config
    site: Site
    events_cal: Mapping[int, tuple[float, str]]
    meta: dict[str, Any] = field(default_factory=dict)


def run(cfg: Config | None = None, site: Site | None = None,
        events_cal: Mapping[int, tuple[float, str]] = EVENT_CAL) -> Result:
    cfg = cfg or Config()
    site = site or build_site()
    rng = np.random.default_rng(cfg.seed)
    start, B = pd.Timestamp(cfg.start_date), cfg.burn_in_days
    days = pd.date_range(start - timedelta(days=B), periods=B + cfg.n_days, freq="D")
    q = np.zeros(len(days))                      # AR(1) traffic quality -> persistent bounce/duration
    for i in range(1, len(days)):
        q[i] = cfg.quality_phi * q[i - 1] + rng.normal(0, cfg.quality_sd)
    lo_adj, hi_adj = cfg.price_adj_bounds

    key_counter = [0]

    def new_key() -> str:                        # bijective scramble of a counter -> unique IDs
        key_counter[0] += 1
        return f"u_{(key_counter[0] * 7919 + 1234567) % 10**7:07d}"

    U_cty: list[str] = []; U_dev: list[str] = []; U_churn: list[bool] = []; U_key: list[str] = []
    day_users: list[list[int]] = []; rows: list[dict] = []; events: list[dict] = []; seq = 0

    for di, d in enumerate(days):
        off = di - B; emit = off >= 0
        mult, label = events_cal.get(off, (1.0, None))
        lam = (cfg.base_sessions * (1 + cfg.growth_per_year * off / 365)
               * DOW_W[d.dayofweek] * season(d) * mult)
        r = cfg.nb_dispersion
        n = int(rng.negative_binomial(r, r / (r + lam)))
        day_users.append([])
        noon = d + timedelta(hours=12); srv = utc_offset_h(cfg.server_tz, noon)
        shift = {c: round(utc_offset_h(v["tz"], noon) - srv) for c, v in COUNTRIES.items()}
        lab = label or ""
        for _ in range(n):
            seq += 1
            is_bot = rng.random() < cfg.bot_share
            known = False
            if is_bot:                           # every bot gets a fresh, never-reused ID
                key, cty, device = new_key(), CTY_CAT(rng), DEV_CAT(rng)
            else:
                uid = None
                if rng.random() < cfg.return_share:
                    if rng.random() < cfg.same_day_share and day_users[di]:
                        pool = day_users[di]; uid = pool[int(rng.integers(len(pool)))]
                    else:
                        dj = di - 1 - int(rng.exponential(cfg.return_lag_mean))
                        if dj >= 0 and day_users[dj]:
                            pool = day_users[dj]; uid = pool[int(rng.integers(len(pool)))]
                if uid is None:                  # new user
                    uid = len(U_key); U_cty.append(CTY_CAT(rng)); U_dev.append(DEV_CAT(rng))
                    U_churn.append(rng.random() < cfg.churn_user_share); U_key.append(new_key())
                else:
                    known = True
                    if U_churn[uid] and rng.random() < 0.5:
                        U_key[uid] = new_key()   # cookie reset
                day_users[di].append(uid)
                key, cty = U_key[uid], U_cty[uid]
                device = U_dev[uid] if rng.random() > cfg.device_switch else DEV_CAT(rng)
            ctry = cty if rng.random() > cfg.missing_geo_rate else "(not set)"
            hour = hour_cat(shift[cty])(rng)
            ts = d + timedelta(hours=hour, minutes=int(rng.integers(60)), seconds=int(rng.integers(60)))

            channel = CH_CAT(rng)
            if "newsletter" in lab and rng.random() < .65: channel = "email"
            if "paid" in lab and rng.random() < .70: channel = "paid_search"
            if "tv_feature" in lab: channel = str(pick(rng, ["direct", "organic_search", "social"], [.5, .3, .2]))
            intent = INTENT_CAT(rng)
            if channel == "email" and rng.random() < .30: intent = "buyer"
            if channel == "social" and rng.random() < .50: intent = "browser"

            if is_bot:
                path = [site.bot_cat(rng)]
            else:
                first = landing(rng, site, channel)
                ftype = site.pages[first]["page_type"]
                z = (-0.35 + CHANNELS[channel]["bounce"] + DEVICES[device]["bounce"]
                     + (-0.45 if known else 0.0)
                     + (0.30 if ftype == "admin" else 0.0) + (-0.20 if ftype == "tour" else 0.0)
                     - 1.0 * q[di] + (2.0 if lab == "site_outage" else 0.0))
                path = [first]
                if rng.random() >= sigmoid(z):
                    cur, step = first, 1
                    while step < 30:
                        nxt = next_page(rng, site, cur, intent, step,
                                        tour_to_funnel=cfg.tour_to_funnel, funnel_step=cfg.funnel_step)
                        if nxt == "exit": break
                        path.append(nxt); cur = nxt; step += 1
            conv = int(path[-1] == FUNNEL[2])    # confirm appears once and IS the conversion
            pv = len(path)

            base = (math.log(26) + 0.78 * math.log(pv) + math.log(CHANNELS[channel]["depth"])
                    + math.log(DEVICES[device]["depth"]) + 0.6 * q[di])
            dur = float(rng.lognormal(base, 0.72))
            if is_bot: dur = 0.0
            elif pv == 1: dur = 0.0 if rng.random() < .85 else min(dur, 25.0)
            dur = round(min(dur, 5400.0), 1)

            tours = [u for u in path if site.pages[u]["page_type"] == "tour"]
            booked = (tours[-1] if tours else site.tour_cat(rng)) if conv else None
            if conv and booked is not None:
                pax = pick(rng, [1, 2, 3, 4, 6], [.18, .52, .12, .13, .05])
                adj = min(max(float(rng.lognormal(0, 0.12)), lo_adj), hi_adj)
                rev = round(site.price(str(booked)) * pax * adj, 2)
            else:
                pax, rev = 0, 0.0
            sid = f"s_{int(rng.integers(1 << 62)):016x}"
            refs = REFERRERS[channel]
            ref_weights = [1.0] * len(refs)
            rows.append({"session_id": sid, "user_id": key, "session_start": ts, "date": d.date(), "hour": hour,
                "dow": int(d.dayofweek), "channel": channel,
                "campaign": CAMPAIGNS(rng) if channel in ("paid_search", "email", "social") else None,
                "referrer": (pick(rng, refs, ref_weights) or "(direct)"),
                "device": device, "browser": BROWSER_BY_DEVICE[device](rng), "country": ctry,
                "language": COUNTRIES[cty]["lang"], "landing_page": path[0], "exit_page": path[-1],
                "pageviews": pv, "duration_sec": dur, "bounced": int(pv == 1),
                "reached_funnel": int(any(u in FUNNEL for u in path)), "converted": conv, "pax": pax,
                "revenue_eur": rev, "booked_tour": booked, "tour_pages_viewed": len(tours),
                "country_pages_viewed": sum(site.pages[u]["page_type"] == "country" for u in path),
                "_seq": seq, "_emit": emit, "_is_bot": int(is_bot), "_anomaly": label, "_true_country": cty,
                "_tzshift": shift[cty]})
            if emit and cfg.write_events:
                rnd_vals = rng.random(pv - 1)
                offs = [0.0] + sorted(float(x) * dur for x in (rnd_vals if isinstance(rnd_vals, np.ndarray) else [rnd_vals])) if pv > 1 else [0.0]
                for i, u in enumerate(path):
                    pg = site.pages[u]; ename = FUNNEL_EVENT.get(u, "page_view")
                    events.append({"event_id": f"{sid}_{i+1:02d}", "session_id": sid, "user_id": key,
                        "event_ts": ts + timedelta(seconds=offs[i]), "event_name": ename, "step": i + 1,
                        "page_url": u, "page_title": pg["page_title"], "section": pg["section"],
                        "region": pg["region"], "page_type": pg["page_type"], "list_price_eur": pg["price"],
                        "channel": channel, "device": device, "country": ctry,
                        "revenue_eur": rev if (conv and ename == "purchase") else 0.0,
                        "_is_bot": int(is_bot)})

    S = pd.DataFrame(rows)
    # returning flag & visit number from chronological history (incl. burn-in)
    S = S.sort_values(["user_id", "session_start", "_seq"], kind="stable")
    S["visit_number"] = S.groupby("user_id").cumcount() + 1
    S["is_returning"] = S["visit_number"] > 1
    S = S[S["_emit"]].sort_values(["session_start", "_seq"]).reset_index(drop=True)
    E = pd.DataFrame(events)
    G = S[["session_id", "date", "_is_bot", "_anomaly", "_true_country", "_tzshift"]].copy()
    S = S.drop(columns=[c for c in S.columns if c.startswith("_")])
    S, E = inject_dirt(S, E, rng, cfg)
    K = daily_kpis(S, G)
    LOG = access_log(E, cfg, rng) if cfg.access_log_lines and len(E) else []
    meta = {"seed": cfg.seed, "config": cfg.as_dict(),
            "rows": {"sessions": len(S), "events": len(E), "ground_truth": len(G)},
            "numpy": np.__version__, "pandas": pd.__version__,
            "python": platform.python_version(), "git_sha": os.environ.get("GIT_SHA")}
    return Result(S, E, G, K, LOG, cfg, site, events_cal, meta)


def inject_dirt(S: pd.DataFrame, E: pd.DataFrame, rng: np.random.Generator,
                cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Duplicate sessions (in both tables) and corrupt some durations."""
    ids = set(S.session_id.sample(frac=cfg.dup_session_rate, random_state=int(rng.integers(1 << 30))))
    S = pd.concat([S, S[S.session_id.isin(ids)]], ignore_index=True)
    if len(E): E = pd.concat([E, E[E.session_id.isin(ids)]], ignore_index=True)
    bad = set(S.session_id.drop_duplicates().sample(frac=cfg.neg_duration_rate,
                                                    random_state=int(rng.integers(1 << 30))))
    S.loc[S.session_id.isin(bad), "duration_sec"] = -1.0
    S = S.sort_values("session_start", kind="stable").reset_index(drop=True)
    if len(E): E = E.sort_values("event_ts", kind="stable").reset_index(drop=True)
    return S, E


def daily_kpis(S: pd.DataFrame, G: pd.DataFrame) -> pd.DataFrame:
    """Daily KPIs on deduplicated human sessions; negative durations excluded from avg."""
    C = S.drop_duplicates("session_id").merge(G[["session_id", "_is_bot"]], on="session_id")
    C = C[C["_is_bot"] == 0]
    k = C.groupby("date").agg(sessions=("session_id", "count"), users=("user_id", "nunique"),
        pageviews=("pageviews", "sum"), bounce_rate=("bounced", "mean"),
        funnel_entry_rate=("reached_funnel", "mean"), bookings=("converted", "sum"),
        revenue_eur=("revenue_eur", "sum")).reset_index()
    dur = C[C.duration_sec >= 0].groupby("date").duration_sec.mean().rename("avg_duration").reset_index()
    k = k.merge(dur, on="date", how="left")
    k["conversion_rate"] = k.bookings / k.sessions
    k["aov_eur"] = (k.revenue_eur / k.bookings.replace(0, np.nan)).round(2)
    return k[["date", "sessions", "users", "pageviews", "bounce_rate", "avg_duration",
              "funnel_entry_rate", "bookings", "revenue_eur", "conversion_rate", "aov_eur"]]


def access_log(E: pd.DataFrame, cfg: Config, rng: np.random.Generator) -> list[str]:
    """Apache combined-format lines sampled from the event table, with real DST offsets."""
    smp = E.drop_duplicates("event_id").sample(n=min(cfg.access_log_lines, len(E)),
                                               random_state=int(rng.integers(1 << 30))).sort_values("event_ts")
    out = []
    for ets, url, dev, isbot in zip(smp.event_ts, smp.page_url, smp.device, smp["_is_bot"]):
        ts = pd.Timestamp(ets).floor("s")
        oh = utc_offset_h(cfg.server_tz, ts)
        tzs = f"{'+' if oh >= 0 else '-'}{int(abs(oh)):02d}{round(abs(oh) % 1 * 60):02d}"
        ip = f"{rng.integers(10,220)}.{rng.integers(0,255)}.{rng.integers(0,255)}.{rng.integers(1,254)}"
        code = pick(rng, [200, 304, 404, 500], [.88, .06, .05, .01])
        ua = UA["bot"] if (isbot and rng.random() < .4) else UA[dev]
        out.append(f'{ip} - - [{ts:%d/%b/%Y:%H:%M:%S} {tzs}] "GET {url} HTTP/1.1" {code} '
                   f'{rng.integers(900,48000)} "-" "{ua}"')
    return out
