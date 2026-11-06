import pytest

from kazner_agents.scenarios import load_scenarios, run_scenario

SCENARIOS = load_scenarios()


def test_there_are_at_least_ten_scenarios():
    assert len(SCENARIOS) >= 10


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s["name"] for s in SCENARIOS])
def test_scenario_meets_its_expectations(scenario):
    _, checks = run_scenario(scenario)
    failed = [f"{name}: {detail}" for name, passed, detail in checks if not passed]
    assert checks and not failed, failed
