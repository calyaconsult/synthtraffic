"""Static model of the simulated website.

Contains the site tree, audience dimensions, seasonality curves and the
journey (Markov) model. Importing this module performs no I/O and draws no
random numbers; the page registry is built explicitly via ``build_site()``.

Structural roles are fixed: the tree must contain the sections ``TOURS``,
``COUNTRY INFO`` and ``ADMIN``. Regions, tours, countries and admin pages
are free to change.
"""
from __future__ import annotations

import bisect
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "BROWSER_BY_DEVICE",
    "BUY",
    "CAMPAIGNS",
    "CHANNELS",
    "CH_CAT",
    "COUNTRIES",
    "CTY_CAT",
    "DEEP",
    "DEVICES",
    "DEV_CAT",
    "DOW_W",
    "EVENT_CAL",
    "FUNNEL",
    "FUNNEL_EVENT",
    "HOME",
    "HOUR_W",
    "INTENT_CAT",
    "MONTH_W",
    "REFERRERS",
    "SITE_TREE",
    "TOUR2COUNTRY",
    "Cat",
    "Site",
    "build_site",
    "hour_cat",
    "landing",
    "next_page",
    "pick",
    "season",
    "sigmoid",
    "utc_offset_h",
]


# --------------------------------------------------------------------- helpers
class Cat:
    """Fast categorical sampler (cumulative weights + bisect). One draw per call."""

    __slots__ = ("cum", "items")

    def __init__(self, items: Sequence[Any], weights: Sequence[float]) -> None:
        self.items = list(items)
        acc: list[float] = []
        c = 0.0
        for w in weights:
            c += float(w)
            acc.append(c)
        self.cum = acc
        if len(self.items) != len(self.cum) or not self.items:
            raise ValueError("Cat needs one weight per item and at least one item")

    def __call__(self, rng: np.random.Generator) -> Any:
        i = bisect.bisect_right(self.cum, rng.random() * self.cum[-1])
        return self.items[min(i, len(self.items) - 1)]


def pick(rng: np.random.Generator, items: Sequence[Any], weights: Sequence[float]) -> Any:
    """One-off weighted choice; consumes exactly one random number."""
    cum: list[float] = []
    c = 0.0
    for w in weights:
        c += float(w)
        cum.append(c)
    return items[min(bisect.bisect_right(cum, rng.random() * cum[-1]), len(items) - 1)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ------------------------------------------------------------------- site tree
# (url, list price EUR or None, popularity weight)
SITE_TREE: Mapping[str, Any] = {
    "TOURS": {
        "MIDDLE EAST": {
            "Oman for Newbies":                 ("/tours/middle-east/oman-for-newbies", 1490, 1.00),
            "Oman, Highlights and Hidden Gems": ("/tours/middle-east/oman-highlights-hidden-gems", 2290, 0.70),
            "Explore Dubai":                    ("/tours/middle-east/explore-dubai", 1190, 1.25),
            "A Visit to the Pharaohs":          ("/tours/middle-east/a-visit-to-the-pharaohs", 1690, 1.10),
            "The Four Imperial Cities":         ("/tours/middle-east/the-four-imperial-cities", 1390, 0.80),
        },
        "EUROPE": {
            "Prague, Kafka's City":             ("/tours/europe/prague-kafkas-city", 790, 0.95),
            "Vienna for Art Lovers":            ("/tours/europe/vienna-for-art-lovers", 890, 0.75),
        },
    },
    "COUNTRY INFO": {
        "MIDDLE EAST": {
            "Oman":    ("/country-info/middle-east/oman", None, 1.00),
            "UAE":     ("/country-info/middle-east/uae", None, 1.15),
            "Egypt":   ("/country-info/middle-east/egypt", None, 1.05),
            "Morocco": ("/country-info/middle-east/morocco", None, 0.85),
        },
        "EUROPE": {
            "Austria":        ("/country-info/europe/austria", None, 0.70),
            "Czech Republic": ("/country-info/europe/czech-republic", None, 0.80),
        },
    },
    "ADMIN": {
        "About":    ("/about", None, 0.55),
        "Feedback": ("/feedback", None, 0.30),
        "Terms":    ("/terms", None, 0.25),
        "Contact":  ("/contact", None, 0.45),
    },
}

# every tour must point to the country page that describes it
TOUR2COUNTRY: Mapping[str, str] = {
    "/tours/middle-east/oman-for-newbies": "/country-info/middle-east/oman",
    "/tours/middle-east/oman-highlights-hidden-gems": "/country-info/middle-east/oman",
    "/tours/middle-east/explore-dubai": "/country-info/middle-east/uae",
    "/tours/middle-east/a-visit-to-the-pharaohs": "/country-info/middle-east/egypt",
    "/tours/middle-east/the-four-imperial-cities": "/country-info/middle-east/morocco",
    "/tours/europe/prague-kafkas-city": "/country-info/europe/czech-republic",
    "/tours/europe/vienna-for-art-lovers": "/country-info/europe/austria",
}

HOME = "/"
FUNNEL = ("/booking/request", "/booking/details", "/booking/confirm")
FUNNEL_EVENT = MappingProxyType({FUNNEL[0]: "begin_booking",
                                 FUNNEL[1]: "add_traveller_details",
                                 FUNNEL[2]: "purchase"})
_SECTIONS = ("TOURS", "COUNTRY INFO")
# click weights on a section hub: (region hubs..., random item, home)
_HUB_TAIL = {"TOURS": (0.22, 0.07), "COUNTRY INFO": (0.24, 0.06)}
_HUB_REGION_2 = {"TOURS": (0.44, 0.27), "COUNTRY INFO": (0.42, 0.28)}


def _region_weights(section: str, n: int) -> tuple[float, ...]:
    if n == 2:                                   # calibrated default
        return _HUB_REGION_2[section]
    total_val: float = float(sum(_HUB_REGION_2[section]))   # same total share, split evenly
    return tuple(total_val / n for _ in range(n))


@dataclass(frozen=True)
class Site:
    pages: Mapping[str, Mapping[str, Any]]
    section_hub: Mapping[str, str]                    # "TOURS" -> "/tours"
    region_hub: Mapping[tuple[str, str], str]         # ("TOURS","EUROPE") -> "/tours/europe"
    region_hubs_of: Mapping[str, tuple[str, ...]]
    hub_weights: Mapping[str, tuple[float, ...]]
    tour_pages: tuple[str, ...]
    country_pages: tuple[str, ...]
    admin_pages: tuple[str, ...]
    tour_cat: Cat
    country_cat: Cat
    admin_cat: Cat
    bot_cat: Cat                                      # bots never enter the funnel
    hub_kids: Mapping[str, Cat]
    tour2country: Mapping[str, str]
    country2tours: Mapping[str, Cat]
    contact_url: str
    about_url: str

    def price(self, url: str) -> float:
        p = self.pages[url]["price"]
        if p is None:
            raise KeyError(f"{url} has no price")
        return float(p)


def build_site(tree: Mapping[str, Any] = SITE_TREE,
               tour2country: Mapping[str, str] = TOUR2COUNTRY) -> Site:
    """Validate a site tree and build the page registry plus samplers."""
    missing = [s for s in (*_SECTIONS, "ADMIN") if s not in tree]
    if missing:
        raise ValueError(f"site tree is missing sections: {missing}")

    pages: dict[str, dict[str, Any]] = {}

    def reg(url, title, section, region=None, price=None, ptype="content", w=1.0):
        if url in pages:
            raise ValueError(f"duplicate URL in site tree: {url}")
        pages[url] = {"url": url, "page_title": title, "section": section, "region": region,
                          "price": price, "page_type": ptype, "weight": float(w)}

    section_hub = {s: "/" + _slug(s) for s in _SECTIONS}
    region_hub = {(s, r): f"{section_hub[s]}/{_slug(r)}" for s in _SECTIONS for r in tree[s]}

    # registration order matters: it fixes sampler order and thus the RNG stream
    reg(HOME, "Home", "HOME", ptype="home")
    for s in _SECTIONS:
        reg(section_hub[s], s.title(), s, ptype="hub")
    for (s, r), u in region_hub.items():
        reg(u, f"{s.title()} - {r.title()}", s, r, ptype="hub")
    for s in _SECTIONS:
        for region, items in tree[s].items():
            for title, (url, price, w) in items.items():
                ptype = "tour" if s == "TOURS" else "country"
                if ptype == "tour" and not (price and price > 0):
                    raise ValueError(f"tour {url} needs a positive price")
                reg(url, title, s, region, price, ptype, w)
    for title, (url, price, w) in tree["ADMIN"].items():
        reg(url, title, "ADMIN", None, None, "admin", w)
    for i, u in enumerate(FUNNEL):
        reg(u, ("Booking Request", "Booking Details", "Booking Confirmation")[i],
            "BOOKING", ptype="funnel")

    def of(t):
        return tuple(u for u, p in pages.items() if p["page_type"] == t)

    def cat_of(urls):
        return Cat(urls, [pages[u]["weight"] for u in urls])

    tours, countries, admins = of("tour"), of("country"), of("admin")
    if not tours or not countries:
        raise ValueError("site tree needs at least one tour and one country page")
    unmapped = [t for t in tours if t not in tour2country]
    bad_target = {t: c for t, c in tour2country.items()
                  if t in pages and pages.get(c, {}).get("page_type") != "country"}
    if unmapped or bad_target:
        raise ValueError(f"tour2country problems: unmapped={unmapped}, bad targets={bad_target}")

    c2t: dict[str, list[str]] = {}
    for t in tours:
        c2t.setdefault(tour2country[t], []).append(t)

    by_title = {pages[u]["page_title"].lower(): u for u in admins}
    ro = MappingProxyType
    return Site(
        pages=ro({u: ro(p) for u, p in pages.items()}),
        section_hub=ro(section_hub),
        region_hub=ro(region_hub),
        region_hubs_of=ro({s: tuple(u for (ss, _), u in region_hub.items() if ss == s)
                           for s in _SECTIONS}),
        hub_weights=ro({s: (*_region_weights(s, len(tree[s])), *_HUB_TAIL[s]) for s in _SECTIONS}),
        tour_pages=tours, country_pages=countries, admin_pages=admins,
        tour_cat=cat_of(tours), country_cat=cat_of(countries),
        admin_cat=cat_of(admins) if admins else Cat([HOME], [1.0]),
        bot_cat=cat_of([u for u, p in pages.items() if p["page_type"] != "funnel"]),
        hub_kids=ro({u: cat_of([v for v, q in pages.items() if q["section"] == s
                                and q["region"] == r and q["page_type"] in ("tour", "country")])
                     for (s, r), u in region_hub.items()}),
        tour2country=ro({t: tour2country[t] for t in tours}),
        country2tours=ro({c: cat_of(v) for c, v in c2t.items()}),
        contact_url=by_title.get("contact", HOME),
        about_url=by_title.get("about", HOME),
    )


# ---------------------------------------------------------- audience dimensions
CHANNELS: Mapping[str, dict[str, float]] = MappingProxyType({
    "organic_search": {"p": .34, "bounce": 0.0, "depth": 1.00},
    "direct":         {"p": .18, "bounce": -.25, "depth": 1.10},
    "paid_search":    {"p": .17, "bounce": .20, "depth": 0.95},
    "social":         {"p": .15, "bounce": .75, "depth": 0.70},
    "email":          {"p": .09, "bounce": -.60, "depth": 1.30},
    "referral":       {"p": .07, "bounce": .05, "depth": 1.00}})
DEVICES: Mapping[str, dict[str, float]] = MappingProxyType({
    "mobile":  {"p": .58, "bounce": .35, "depth": .80},
    "desktop": {"p": .33, "bounce": -.20, "depth": 1.25},
    "tablet":  {"p": .09, "bounce": .05, "depth": 1.00}})
# real IANA zones -> correct local hours incl. DST (US 1 Nov, EU 25 Oct, Gulf never)
COUNTRIES: Mapping[str, dict[str, Any]] = MappingProxyType({
    "CH": {"p": .33, "tz": "Europe/Zurich", "lang": "de-CH"},
    "DE": {"p": .22, "tz": "Europe/Berlin", "lang": "de-DE"},
    "AT": {"p": .09, "tz": "Europe/Vienna", "lang": "de-AT"},
    "FR": {"p": .08, "tz": "Europe/Paris", "lang": "fr-FR"},
    "GB": {"p": .09, "tz": "Europe/London", "lang": "en-GB"},
    "US": {"p": .11, "tz": "America/New_York", "lang": "en-US"},
    "AE": {"p": .04, "tz": "Asia/Dubai", "lang": "en-AE"},
    "OM": {"p": .04, "tz": "Asia/Muscat", "lang": "en-OM"}})
BROWSER_BY_DEVICE = MappingProxyType({
    "mobile":  Cat(["Chrome", "Safari", "Samsung Internet", "Firefox"], [.42, .34, .18, .06]),
    "desktop": Cat(["Chrome", "Edge", "Safari", "Firefox"], [.58, .18, .12, .12]),
    "tablet":  Cat(["Safari", "Chrome", "Firefox"], [.62, .30, .08])})
CAMPAIGNS = Cat(["oman_autumn", "dubai_citybreak", "egypt_nile", "morocco_imperial",
                 "prague_weekend", "vienna_art", "brand_generic", "newsletter_oct"], [1] * 8)
REFERRERS: Mapping[str, Sequence[str | None]] = MappingProxyType({
    "organic_search": ["https://www.google.com/", "https://www.bing.com/", "https://duckduckgo.com/"],
    "social": ["https://www.instagram.com/", "https://l.facebook.com/", "https://t.co/",
               "https://www.pinterest.ch/"],
    "referral": ["https://www.tripadvisor.ch/", "https://www.reddit.com/r/travel/",
                 "https://blog.wanderlust.example/"],
    "email": ["https://mail.google.com/"], "paid_search": ["https://www.google.com/aclk"],
    "direct": [None]})
CH_CAT = Cat(list(CHANNELS), [v["p"] for v in CHANNELS.values()])
DEV_CAT = Cat(list(DEVICES), [v["p"] for v in DEVICES.values()])
CTY_CAT = Cat(list(COUNTRIES), [float(v["p"]) for v in COUNTRIES.values()])
INTENT_CAT = Cat(["browser", "researcher", "buyer", "admin_seeker"], [.46, .34, .14, .06])
BUY = MappingProxyType({"buyer": 1.0, "researcher": .35, "browser": .10, "admin_seeker": .05})
DEEP = MappingProxyType({"buyer": .80, "researcher": .86, "browser": .58, "admin_seeker": .35})


# ------------------------------------------------------ seasonality (leisure)
DOW_W = MappingProxyType({0: 1.00, 1: 1.02, 2: 1.00, 3: 0.98, 4: 0.86, 5: 1.08, 6: 1.32})
HOUR_W = (0.30, 0.18, 0.11, 0.08, 0.07, 0.09, 0.18, 0.38, 0.60, 0.74, 0.82, 0.86,
          0.84, 0.80, 0.78, 0.80, 0.86, 0.94, 1.00, 1.12, 1.20, 1.10, 0.78, 0.50)   # LOCAL time
MONTH_W = (1.18, 1.12, 1.05, .98, .92, .86, .80, .84, 1.02, 1.14, 1.10, .95)
_MID_DOY = (15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349)
_SX = np.array(_MID_DOY + (_MID_DOY[0] + 365,))
_SY = np.array(MONTH_W + (MONTH_W[0],))
# day offset from start -> (traffic multiplier, label)
EVENT_CAL = MappingProxyType({21: (2.9, "newsletter_blast"), 22: (1.8, "newsletter_blast_tail"),
                              47: (0.15, "site_outage"), 63: (3.4, "tv_feature_oman"),
                              64: (2.1, "tv_feature_tail"), 80: (2.2, "paid_campaign_push")})


def season(d: pd.Timestamp) -> float:
    """Smooth month-of-year multiplier (interpolated between mid-month anchors)."""
    doy = d.dayofyear if d.dayofyear >= _MID_DOY[0] else d.dayofyear + 365
    return float(np.interp(doy, _SX, _SY))


@cache
def utc_offset_h(tzname: str, ts: pd.Timestamp) -> float:
    """UTC offset in hours for a naive local timestamp in ``tzname``."""
    return pd.Timestamp(ts).tz_localize(tzname, ambiguous=False,
                                        nonexistent="shift_forward").utcoffset().total_seconds() / 3600


@cache
def hour_cat(shift: int) -> Cat:  # type: ignore[misc]
    """Server-hour sampler for visitors whose local time is server time + ``shift``."""
    weights: list[float] = [float(HOUR_W[(h + shift) % 24]) for h in range(24)]
    return Cat(list(range(24)), weights)


# --------------------------------------------------------------- journey model
def next_page(rng: np.random.Generator, site: Site, cur: str, intent: str, step: int, *,
              tour_to_funnel: float, funnel_step: Sequence[float]) -> str:
    """Next URL after ``cur``, or ``"exit"``. Order of random draws is part of the contract."""
    p = site.pages[cur]
    t = p["page_type"]
    buy = BUY[intent]
    if t == "funnel":                      # explicit progression, no fatigue
        i = FUNNEL.index(cur)
        if i < 2 and rng.random() < funnel_step[i] * (0.75 + 0.35 * buy):
            return FUNNEL[i + 1]
        if i < 2 and rng.random() < .45:
            return site.tour_cat(rng)
        return "exit"                      # after confirm the walk ends
    if rng.random() > DEEP[intent] * max(0.25, 1 - 0.045 * step):
        return "exit"
    tours_hub, info_hub = site.section_hub["TOURS"], site.section_hub["COUNTRY INFO"]
    if t == "home":
        return pick(rng, [tours_hub, info_hub, site.tour_cat(rng), site.admin_cat(rng)],
                    [.46, .24, .22, .08])
    if cur == tours_hub:
        return pick(rng, [*site.region_hubs_of["TOURS"], site.tour_cat(rng), HOME],
                    site.hub_weights["TOURS"])
    if cur == info_hub:
        return pick(rng, [*site.region_hubs_of["COUNTRY INFO"], site.country_cat(rng), HOME],
                    site.hub_weights["COUNTRY INFO"])
    if t == "hub":
        return pick(rng, [site.hub_kids[cur](rng), site.section_hub[p["section"]], HOME],
                    [.82, .12, .06])
    if t == "tour":
        return pick(rng, [FUNNEL[0], site.tour2country[cur], site.region_hub[("TOURS", p["region"])],
                          site.tour_cat(rng), site.contact_url, HOME],
                    [tour_to_funnel * buy, .24, .17, .24, .08, .06])
    if t == "country":
        rel = site.country2tours[cur](rng) if cur in site.country2tours else site.tour_cat(rng)
        return pick(rng, [rel, site.region_hub[("COUNTRY INFO", p["region"])], site.country_cat(rng),
                          info_hub, HOME],
                    [.34 + .2 * buy, .19, .22, .13, .08])
    if t == "admin":
        return pick(rng, [HOME, tours_hub, site.admin_cat(rng), site.tour_cat(rng)],
                    [.38, .22, .22, .18])
    return "exit"


def landing(rng: np.random.Generator, site: Site, channel: str) -> str:
    """Landing page for a session arriving via ``channel``."""
    if channel == "organic_search":
        return pick(rng, [site.country_cat(rng), site.tour_cat(rng), HOME, site.section_hub["TOURS"]],
                    [.42, .34, .14, .10])
    if channel == "paid_search":
        return pick(rng, [site.tour_cat(rng), site.section_hub["TOURS"], HOME], [.70, .18, .12])
    if channel == "social":
        return pick(rng, [site.tour_cat(rng), site.country_cat(rng), HOME], [.55, .30, .15])
    if channel == "email":
        return pick(rng, [site.tour_cat(rng), HOME, site.section_hub["TOURS"]], [.68, .16, .16])
    if channel == "direct":
        return pick(rng, [HOME, site.tour_cat(rng), site.contact_url], [.62, .28, .10])
    return pick(rng, [site.country_cat(rng), site.tour_cat(rng), HOME, site.about_url],
                [.40, .33, .17, .10])
