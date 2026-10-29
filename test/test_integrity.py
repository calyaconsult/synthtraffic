# tests/test_integrity.py
import pytest
from synthtraffic import config, generate, validate

@pytest.fixture(scope="session")
def result():
    return generate.run(config.Config(seed=42, n_days=35, burn_in_days=28))

def test_all_integrity_checks(result):
    T, _ = validate.run_tests(result)
    failed = T[~T.passed]
    assert failed.empty, "\n" + failed.to_string(index=False)

@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42, 123, 2026])
def test_kpi_targets_multi_seed(seed):
    res = generate.run(config.Config(seed=seed))
    T, _ = validate.run_tests(res)
    assert T[~T.passed].empty
