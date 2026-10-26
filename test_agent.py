"""Guardrail tests. Run with:  python test_agent.py   (or: pytest -q)"""
import numpy as np
import pandas as pd

from agent import AgentConfig, BudgetAgent
from evaluate import run_episode
from simulator import CHANNELS, generate_baseline_csv

CFG = AgentConfig()
TOL = 0.011  # one cent of rounding slack


def _check_path(budgets_df, cfg):
    arr = budgets_df[CHANNELS].to_numpy()
    total = cfg.total_budget
    assert np.allclose(arr.sum(axis=1), total, atol=TOL), "total daily budget must stay fixed"
    assert (arr >= cfg.min_share * total - TOL).all(), "floor violated (a channel went below its minimum)"
    assert (arr > 0).all(), "no channel may ever hit 0"
    assert (arr <= cfg.max_share * total + TOL).all(), "ceiling violated"
    steps = np.abs(arr[1:] / arr[:-1] - 1)
    assert (steps <= cfg.max_daily_change + 1e-3).all(), "daily change exceeded +/-20%"


def test_guardrails_hold_across_seeds():
    for seed in range(25):
        _, budgets, _ = run_episode("agent", seed, CFG)
        _check_path(budgets, CFG)


def test_guardrails_hold_for_clicks_objective():
    cfg = AgentConfig(objective="clicks")
    for seed in range(10):
        _, budgets, _ = run_episode("agent", seed, cfg)
        _check_path(budgets, cfg)


def test_every_decision_has_a_reason():
    _, _, decisions = run_episode("agent", 0, CFG)
    assert len(decisions) == 30 * len(CHANNELS)
    assert all(isinstance(d["reason"], str) and len(d["reason"]) > 20 for d in decisions)


def test_warmup_is_equal_split():
    agent = BudgetAgent()
    budgets, _ = agent.propose(pd.DataFrame(columns=["date", "channel", "spend", "impressions", "clicks", "conversions"]).astype(
        {"date": "datetime64[ns]"}), pd.Timestamp("2026-09-25"))
    assert len(set(round(v, 2) for v in budgets.values())) == 1


def test_csv_has_required_shape(tmp_path):
    df = generate_baseline_csv(str(tmp_path / "d.csv"), seed=1, days=30)
    assert list(df.columns) == ["date", "channel", "spend", "impressions", "clicks", "conversions"]
    assert len(df) == 30 * 3 and df["channel"].nunique() == 3


if __name__ == "__main__":
    import pathlib, tempfile
    test_guardrails_hold_across_seeds()
    test_guardrails_hold_for_clicks_objective()
    test_every_decision_has_a_reason()
    test_warmup_is_equal_split()
    test_csv_has_required_shape(pathlib.Path(tempfile.mkdtemp()))
    print("All guardrail tests passed.")
