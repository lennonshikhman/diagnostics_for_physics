from diagnostic_suite.self_check import run_self_check


def test_deterministic_self_check() -> None:
    values = run_self_check()
    assert values["exact_semigroup"] < 1.0e-6
    assert values["broken_semigroup"] > 1.0e-3
