from __future__ import annotations
import pytest
from synthtraffic import Config, load, run, run_tests

FAST = Config(n_days=35, access_log_lines=300)

@pytest.fixture(scope="session")
def fast_result():
    return run(FAST)

def test_integrity_checks_pass(fast_result):
    results, _ = run_tests(fast_result, kpi_checks=False)
    failed = results[~results.passed]
    assert failed.empty, "\n" + failed.to_string(index=False)

def test_no_label_leakage(fast_result):
    from synthtraffic import export_frames
    sessions, events, gt = export_frames(fast_result)
    assert not [c for c in sessions.columns if "bot" in c or "anomaly" in c]
    assert set(gt.columns) == {"session_id", "date", "is_bot", "anomaly"}

def test_reproducible_for_same_seed():
    a, b = run(Config(n_days=14)), run(Config(n_days=14))
    assert a.sessions.equals(b.sessions)
    assert a.events.equals(b.events)

def test_different_seeds_differ():
    a, b = run(Config(n_days=14, seed=1)), run(Config(n_days=14, seed=2))
    assert not a.sessions.equals(b.sessions)

def test_events_can_be_disabled():
    res = run(Config(n_days=14, write_events=False, access_log_lines=0))
    assert len(res.events) == 0 and res.log == []
    results, _ = run_tests(res, kpi_checks=False)
    assert results[~results.passed].empty

@pytest.mark.parametrize("kwargs", [
    {"n_days": 1}, {"bot_share": 1.5}, {"quality_phi": 1.0},
    {"price_adj_bounds": (1.2, 1.5)}, {"nb_dispersion": 0},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)

def test_unknown_config_key_rejected():
    with pytest.raises(ValueError, match="unknown config keys"):
        load(None, does_not_exist=1)

def test_write_outputs(tmp_path, fast_result):
    from synthtraffic import write_outputs
    paths = write_outputs(fast_result, tmp_path)
    names = {p.name for p in paths}
    assert {"sessions.csv", "daily_kpis.csv", "_ground_truth.csv",
            "run_metadata.json"} <= names
    assert all(p.stat().st_size > 0 for p in paths)

def test_cli_dry_run(capsys):
    from synthtraffic.cli import main
    assert main(["--n-days", "14", "--no-events", "--dry-run"]) == 0
    assert "dry run" in capsys.readouterr().out

# --- statistical validation: needs the full window, so marked slow ---------
@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42, 123, 2026])
def test_kpi_targets(seed):
    results, _ = run_tests(run(Config(seed=seed)))
    failed = results[~results.passed]
    assert failed.empty, "\n" + failed.to_string(index=False)

@pytest.mark.slow
def test_weekly_rhythm_on_average():
    """Per-seed ACF(7) ranges 0.30-0.70; the mean is the stable quantity."""
    import numpy as np
    from synthtraffic import kpi_metrics
    from synthtraffic.validate import anomaly_dates
    vals = []
    for seed in range(8):
        cfg = Config(seed=seed)
        res = run(cfg)
        vals.append(kpi_metrics(res.kpi, anomaly_dates(cfg))["acf7_detrended"])
    assert np.mean(vals) >= 0.45, f"mean ACF(7) too low: {np.mean(vals):.2f}"
