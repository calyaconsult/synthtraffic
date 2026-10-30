# synthtraffic

Synthetic web-traffic data for a travel / tour-operator website: sessions, page-view
and booking events, daily KPIs, an Nginx-style access log and a ground-truth answer key.
The data is **rule-based** (no real user data involved) and **validated**: every run can
be checked against 26 integrity and statistical tests.

Use it to test analytics pipelines, dashboards, anomaly and bot detection, attribution
models or SQL exercises without touching personal data.

```text
HOME
├── TOURS
│   ├── MIDDLE EAST: Oman for Newbies · Oman, Highlights and Hidden Gems · Explore Dubai
│   │                A Visit to the Pharaohs · The Four Imperial Cities
│   └── EUROPE:      Prague, Kafka's City · Vienna for Art Lovers
├── COUNTRY INFO
│   ├── MIDDLE EAST: Oman · UAE · Egypt · Morocco
│   └── EUROPE:      Austria · Czech Republic
├── ADMIN:           About · Feedback · Terms · Contact
└── BOOKING funnel:  /booking/request → /booking/details → /booking/confirm
```

27 pages in total, including section and region hub pages generated from the tree.

---

## Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/calyaconsult/synthtraffic.git
cd synthtraffic
pip install -e .              # core: numpy, pandas
pip install -e ".[yaml]"      # + YAML config files
pip install -e ".[parquet]"   # + Parquet output
pip install -e ".[dev]"       # + pytest, ruff, mypy, pyyaml, pyarrow
```

## Quick start

### Command line

```bash
synthtraffic                                  # default dataset -> outputs/
synthtraffic --validate                       # + run all checks, write test_report.txt
synthtraffic --seed 7 --n-days 180 --format parquet --out-dir data/
synthtraffic --config configs/small.yaml --dry-run
synthtraffic --seeds 20 --integrity-only      # validate 20 seeds, write nothing
python -m synthtraffic --help                 # works without the console script
```

| Exit code | Meaning |
|---|---|
| 0 | success |
| 1 | at least one check failed (files are still written so you can inspect them) |
| 2 | invalid arguments or configuration |

Messages go to stderr, results to stdout. `-v` shows timings and file sizes, `-vv` adds debug output.

### Python

```python
from synthtraffic import Config, run, run_tests, report, write_outputs

res = run(Config(seed=42))
results, metrics = run_tests(res)          # DataFrame with one row per check
print(report(results, metrics, seed=42))
write_outputs(res, "outputs")              # or fmt="parquet"

res.sessions, res.events, res.kpi          # pandas DataFrames
```

Importing the package has no side effects: no I/O, no random draws.

---

## Output files

| File | Contents |
|---|---|
| `sessions.csv` | one row per session (plus deliberate duplicates, see below) |
| `events.csv` | one row per page view / funnel step |
| `events_sample.jsonl` | first 3,000 events as JSON lines |
| `daily_kpis.csv` | sessions, users, pageviews, bounce rate, avg. duration, funnel entry rate, bookings, revenue, conversion rate, AOV per day (bots and duplicates excluded) |
| `access.log` | sampled requests in combined log format, with correct DST offsets for `server_tz` |
| `_ground_truth.csv` | answer key: `session_id, date, is_bot, anomaly` |
| `test_report.txt` | check results (only with `--validate`) |
| `run_metadata.json` | seed, full config, Python/NumPy/pandas versions, `GIT_SHA` env var, timestamp, file list |

With seed 42 and the defaults (91 days from 2026-08-01) you get about 56,300 sessions,
114,700 events, 1,097 bookings and EUR 3.63 M revenue.

### Main columns

**sessions:** `session_id, user_id, session_start, date, hour, dow, channel, campaign,
referrer, device, browser, country, language, landing_page, exit_page, pageviews,
duration_sec, bounced, reached_funnel, converted, pax, revenue_eur, booked_tour,
tour_pages_viewed, country_pages_viewed, visit_number, is_returning`

**events:** `event_id, session_id, user_id, event_ts, event_name, step, page_url,
page_title, section, region, page_type, list_price_eur, channel, device, country, revenue_eur`

`event_name` is `page_view`, `begin_booking`, `add_traveller_details` or `purchase`.
Times are in server local time (`server_tz`, default `Europe/Zurich`).

### Deliberate imperfections

Real analytics data is messy, so this data is too. Your pipeline should cope with:

| Imperfection | Default rate |
|---|---|
| Bot sessions (single page, 0 s), **not flagged** in sessions/events | 2.5 % |
| Duplicated sessions, consistently duplicated in events | 0.2 % |
| `duration_sec = -1` (tracking error) | 0.4 % |
| `country = "(not set)"` | 3.5 % |
| `referrer = "(direct)"` | all direct traffic |
| Cookie resets: one person appears under several `user_id`s | 12 % of users can be affected |
| Anomaly days: newsletter blast, site outage, TV feature, paid campaign | 6 days |

The labels for bots and anomalies live **only** in `_ground_truth.csv`. Use it to
score detection methods, not as a model input. Bot sessions on anomaly days carry both
labels, so count them separately when you evaluate bot vs. anomaly detection.
Set `sparse_ground_truth=False` to get a row for every session; otherwise sessions not
listed are normal human visits.

---

## What the model simulates

- **Volume:** negative-binomial daily counts × trend × day-of-week × smooth seasonal
  curve × event calendar. Leisure-travel pattern: Sunday strongest, Friday weakest.
- **Time of day:** an hourly curve in **visitor local time** (peak ~20:00), converted to
  server time per country through real IANA time zones, including the different DST
  switch dates in Europe, the US and the Gulf.
- **Users:** returning visits clustered in time (same day or ~1 + Exp(6) days later),
  28 days of burn-in history so `is_returning` is realistic from day one, cookie churn,
  device switching. `visit_number` and `is_returning` are derived from each user's
  history, never drawn at random.
- **Journeys:** a Markov model over the site tree, driven by channel, device and visitor
  intent (browser, researcher, buyer, admin seeker). Tours link to their country page and
  vice versa; the booking funnel has explicit step probabilities.
- **Day-level quality:** an AR(1) factor, so bounce rate and session duration persist
  from day to day instead of being pure noise.
- **Revenue:** list price × travellers × bounded noise (−25 % discount to +35 % upsell).

## Configuration

All fields of `Config` (frozen dataclass, validated on creation). Precedence:
**defaults ← YAML file ← command-line / keyword overrides.** Unknown keys are rejected,
and all errors are reported in one message.

| Field | Default | Meaning |
|---|---|---|
| `seed` | 42 | random seed |
| `start_date` | `2026-08-01` | first exported day |
| `n_days` | 91 | exported days (≥ 14) |
| `burn_in_days` | 28 | simulated history before `start_date`, not exported |
| `base_sessions` | 520 | average sessions on a neutral weekday |
| `growth_per_year` | 0.55 | linear trend |
| `nb_dispersion` | 120 | negative-binomial *r*; lower = noisier |
| `return_share` | 0.38 | share of human sessions from known users |
| `same_day_share` | 0.45 | of those: revisit on the same day |
| `return_lag_mean` | 6.0 | otherwise: lag in days ≈ 1 + Exp(mean) |
| `churn_user_share` | 0.12 | users whose cookie resets now and then |
| `device_switch` | 0.15 | chance a session uses another device |
| `quality_phi`, `quality_sd` | 0.75, 0.12 | AR(1) day-level traffic quality |
| `tour_to_funnel` | 0.22 | tour page → booking request weight (× buyer intent) |
| `funnel_step` | (0.72, 0.80) | request → details, details → confirm |
| `price_adj_bounds` | (0.75, 1.35) | bounds for the price noise |
| `bot_share` | 0.025 | share of bot sessions |
| `missing_geo_rate` | 0.035 | share of `(not set)` countries |
| `dup_session_rate` | 0.002 | duplicated sessions |
| `neg_duration_rate` | 0.004 | sessions with `duration_sec = -1` |
| `sparse_ground_truth` | true | ground truth lists labelled sessions only |
| `write_events` | true | generate events, JSONL sample and access log |
| `access_log_lines` | 1200 | lines in `access.log` (0 = none) |
| `server_tz` | `Europe/Zurich` | server time zone (IANA) |
| `out_dir` | `outputs` | output directory |

Example `configs/small.yaml`:

```yaml
n_days: 14
access_log_lines: 200
funnel_step: [0.72, 0.80]
```

Seasonality curves (`DOW_W`, `HOUR_W`, `MONTH_W`), channel / device / country mixes
and journey weights are module constants in `synthtraffic/model.py`.

## Custom site tree and event calendar

The site tree is plain data: `{section: {region: {title: (url, price_eur, weight)}}}`.
The sections `TOURS`, `COUNTRY INFO` and `ADMIN` are fixed roles; regions, tours,
countries and admin pages are free. Hub URLs are derived from the names
(`"MIDDLE EAST"` → `/tours/middle-east`). Every tour needs a positive price and a
country page in `tour2country`.

```python
from copy import deepcopy
from synthtraffic import Config, SITE_TREE, TOUR2COUNTRY, EVENT_CAL, build_site, run

tree = deepcopy(dict(SITE_TREE))
tree["TOURS"] = dict(tree["TOURS"], ASIA={
    "Kyoto in Autumn": ("/tours/asia/kyoto-in-autumn", 2490, 0.9)})
tree["COUNTRY INFO"] = dict(tree["COUNTRY INFO"], ASIA={
    "Japan": ("/country-info/asia/japan", None, 0.9)})
t2c = dict(TOUR2COUNTRY, **{"/tours/asia/kyoto-in-autumn": "/country-info/asia/japan"})

site = build_site(tree, t2c)          # raises ValueError on an inconsistent tree

# day offset from start_date -> (traffic multiplier, label)
calendar = dict(EVENT_CAL, **{30: (2.5, "kyoto_launch")})

res = run(Config(), site=site, events_cal=calendar)
```

---

## Validation

`run_tests(res, kpi_checks=True)` returns one row per check.

**Integrity checks (16)** hold for any seed and any window length, e.g.:

- revenue in events equals revenue in sessions; exactly one `purchase` per booking
- pageviews equal the number of event rows; the confirm page appears at most once
- `is_returning == (visit_number > 1)`, visit numbers chronological without gaps
- bot IDs never reused or shared with humans
- local-time traffic profile and `access.log` DST offsets correct (via `zoneinfo`)
- duplicates consistent across tables; no label columns in sessions/events
- browser matches device; bounces have ~0 s duration; duration grows with depth
- revenue = price × pax × bounded noise; missing-geo share 2–5 %

**Statistical checks (10 KPI bands)** need about the full 91-day window. The bands were
fixed before tuning:

| KPI | Band | Seed 42 |
|---|---|---|
| Conversion rate | 1.5–2.5 % | 2.00 % |
| Funnel completion | 40–60 % | 56 % |
| Days without bookings | ≤ 5 % | 1.1 % |
| Weekly rhythm, detrended ACF(7) | ≥ 0.45 | 0.65 |
| Residual noise (CV) | 6–12 % | 10.0 % |
| Weekend / weekday traffic | 1.05–1.30 | 1.22 |
| Bounce rate ACF(1) / SD | ≥ 0.20 / 0.02–0.06 | 0.42 / 0.033 |
| Duration ACF(1) | ≥ 0.20 | 0.49 |
| Sessions per user per day | 1.10–1.50 | 1.20 |

Labelled anomaly days are excluded from the KPI statistics. Below 91 days the CLI
switches to integrity checks automatically (with a warning), and the 19–21 h local-peak
check is replaced by a correlation with the hourly profile.

---

## Known limitations

Please read these before relying on the data.

- **Weekly rhythm depends on the seed.** With 91 days there are only ~13 weekly cycles,
  so ACF(7) ranges from about 0.30 to 0.70 across seeds. In a 20-seed run, seeds 15 and
  1000 missed the ≥ 0.45 band. The slow test suite therefore uses known-good seeds
  (1, 7, 42, 123, 2026) **plus** a test on the mean across seeds. The bands were not
  loosened to hide this.
- **Rule-based, not fitted.** Distributions and weights are plausible assumptions, not
  estimated from real data. Treat absolute numbers as illustrative.
- **More than two regions per section** have no calibrated hub weights: the existing share
  is split evenly.
- **Missing `Contact` / `About` pages:** those clicks go to the home page.
- **TV-feature days** triple traffic but drop the conversion rate to ~1.2 % because of the
  shift to social and direct traffic. This is intended.
- **Ambiguous DST hour:** log lines in the repeated hour at the end of DST cannot be
  checked and are skipped (and counted) by the DST test.
- **Reproducibility** is exact for the same seed **and** the same NumPy/pandas versions.
  `run_metadata.json` records the versions; pin them for byte-identical reruns.
  Changing the order of random draws anywhere in the code changes the output.

## Privacy

No real user data is used or reproduced. IPs, IDs and user agents are synthetic, so the
data can be shared freely without GDPR / revDSG concerns.

---

## Development

```bash
pip install -e ".[dev]"
ruff check src tests
mypy src
pytest -m "not slow"     # fast: integrity, reproducibility, config, CLI
pytest -m slow           # full 91-day runs on 5 seeds + mean ACF(7) over 8 seeds
```

CI (`.github/workflows/ci.yml`) runs lint, type checks and the fast tests on every push
and pull request (Python 3.10 and 3.12), and the slow suite weekly and on tags.

### Project layout

```text
src/synthtraffic/
├── __init__.py     public API, __version__
├── __main__.py     python -m synthtraffic
├── cli.py          command-line interface
├── config.py       Config dataclass, YAML/override loader
├── model.py        site tree, audience, seasonality, journey model
├── generate.py     simulation, dirt injection, daily KPIs, access log
├── export.py       label stripping, file writing, run metadata
└── validate.py     KPI metrics, 26 checks, text report
tests/test_suite.py
```

### History

- **v2.0.0:** package version. Fixes from the v1 audit: time-zone shift direction,
  double-counted booking confirmations, user-history-based returning flag, leisure
  seasonality, sparse conversions, label leakage, inconsistent duplicates, DST in logs,
  bounded price noise; adds the validation suite and CLI.
- **v1:** single Colab script.

## License

MIT, see [LICENSE](LICENSE).
