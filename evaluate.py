"""
evaluate.py - Score the agent against baselines in a closed loop.

For each random seed we replay the same 30-day market three ways:
  * baseline : fixed equal split (Search / Social / Display each get 1/3)
  * agent    : BudgetAgent re-allocates every day using only past data
  * oracle   : hindsight upper bound that knows the true response curves
               (same guardrail floor/ceiling; not achievable in practice)

Reports total conversions, CPA and the paired uplift of agent vs baseline.

Usage:
    python evaluate.py                # 100 seeds
    python evaluate.py --seeds 20 --objective clicks
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from agent import AgentConfig, BudgetAgent, empty_history, write_decision_log
from simulator import (CHANNELS, N_DAYS, SHIFT_DAY, START_DATE, TOTAL_DAILY_BUDGET,
                       AdEnvironment, expected_conversions)


def run_episode(strategy: str, seed: int, config: AgentConfig):
    """Simulate N_DAYS under `strategy` ('agent' or 'baseline'). Returns history, budgets, decisions."""
    env, agent = AdEnvironment(seed), BudgetAgent(config=config)
    history, prev, budget_rows, decisions_all = empty_history(), None, [], []
    for d in range(N_DAYS):
        date = START_DATE + pd.Timedelta(days=d)
        if strategy == "agent":
            budgets, decisions = agent.propose(history, date, prev)
            decisions_all += decisions
        else:
            budgets = agent.equal_split()
        history = pd.concat([history, env.step(date, d, budgets)], ignore_index=True)
        budget_rows.append({"day": d, "date": date, **budgets})
        prev = budgets
    return history, pd.DataFrame(budget_rows), decisions_all


def expected_total(budgets_df: pd.DataFrame) -> np.ndarray:
    """Noise-free expected conversions per day for a budget path (on the true curves)."""
    out = []
    for _, row in budgets_df.iterrows():
        out.append(sum(float(expected_conversions(ch, row[ch], int(row["day"]), row["date"])) for ch in CHANNELS))
    return np.array(out)


def oracle_daily(config: AgentConfig) -> np.ndarray:
    """Best possible expected conversions per day (grid search, $10 steps, same floor/ceiling)."""
    lo, hi, total = config.min_share * config.total_budget, config.max_share * config.total_budget, config.total_budget
    grid = np.arange(lo, hi + 1e-9, 10.0)
    a, b = np.meshgrid(grid, grid, indexing="ij")
    c = total - a - b
    ok = (c >= lo - 1e-9) & (c <= hi + 1e-9)
    out = []
    for d in range(N_DAYS):
        date = START_DATE + pd.Timedelta(days=d)
        conv = (expected_conversions(CHANNELS[0], a, d, date) + expected_conversions(CHANNELS[1], b, d, date)
                + expected_conversions(CHANNELS[2], np.clip(c, 0, None), d, date))
        out.append(float(np.where(ok, conv, -np.inf).max()))
    return np.array(out)


def main():
    p = argparse.ArgumentParser(description="Evaluate the ad optimization agent vs an equal-split baseline.")
    p.add_argument("--seeds", type=int, default=100)
    p.add_argument("--demo-seed", type=int, default=42, help="seed whose decisions are written to the log")
    p.add_argument("--objective", choices=["conversions", "clicks"], default="conversions")
    p.add_argument("--outdir", default="results")
    p.add_argument("--log", default="logs/decisions.csv")
    args = p.parse_args()
    cfg = AgentConfig(objective=args.objective)
    col = "conversions" if args.objective == "conversions" else "clicks"
    os.makedirs(args.outdir, exist_ok=True)

    per_seed, cum_agent, cum_base = [], [], []
    for seed in range(args.seeds):
        h_a, b_a, dec = run_episode("agent", seed, cfg)
        h_b, b_b, _ = run_episode("baseline", seed, cfg)
        daily_a = h_a.groupby("date")[col].sum().to_numpy()
        daily_b = h_b.groupby("date")[col].sum().to_numpy()
        per_seed.append({
            "seed": seed, "spend": h_a["spend"].sum(),
            "agent_total": daily_a.sum(), "baseline_total": daily_b.sum(),
            "agent_after_shift": daily_a[SHIFT_DAY:].sum(), "baseline_after_shift": daily_b[SHIFT_DAY:].sum(),
            "agent_expected": expected_total(b_a).sum(), "baseline_expected": expected_total(b_b).sum(),
        })
        cum_agent.append(np.cumsum(daily_a)); cum_base.append(np.cumsum(daily_b))
        if seed == args.demo_seed:                                  # save one run's decision log + budgets
            write_decision_log(dec, args.log, append=False)
            b_a.assign(strategy="agent").to_csv(f"{args.outdir}/demo_budgets_agent.csv", index=False)
            h_a.to_csv(f"{args.outdir}/demo_history_agent.csv", index=False)

    df = pd.DataFrame(per_seed)
    df.to_csv(f"{args.outdir}/per_seed.csv", index=False)
    oracle = oracle_daily(cfg)
    pd.DataFrame({"day": range(N_DAYS), "agent": np.mean(cum_agent, axis=0), "baseline": np.mean(cum_base, axis=0),
                  "oracle_expected": np.cumsum(oracle)}).to_csv(f"{args.outdir}/cumulative_mean.csv", index=False)

    n, spend = len(df), df["spend"].iloc[0]
    diff = df["agent_total"] - df["baseline_total"]
    ci = 1.96 * diff.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    summary = {
        "objective": args.objective, "seeds": n, "days": N_DAYS, "total_spend_per_run": float(spend),
        "baseline_total_mean": float(df["baseline_total"].mean()), "baseline_total_std": float(df["baseline_total"].std()),
        "agent_total_mean": float(df["agent_total"].mean()), "agent_total_std": float(df["agent_total"].std()),
        "oracle_expected_total": float(oracle.sum()),
        "uplift_mean": float(diff.mean()), "uplift_ci95": float(ci),
        "uplift_pct": float(diff.mean() / df["baseline_total"].mean() * 100),
        "win_rate_vs_baseline": float((diff > 0).mean()),
        "baseline_cost_per_result": float(spend / df["baseline_total"].mean()),
        "agent_cost_per_result": float(spend / df["agent_total"].mean()),
        "uplift_pct_after_shift": float((df["agent_after_shift"].mean() / df["baseline_after_shift"].mean() - 1) * 100),
        "share_of_oracle_gap_closed_pct": float(
            (df["agent_expected"].mean() - df["baseline_expected"].mean())
            / (oracle.sum() - df["baseline_expected"].mean()) * 100),
    }
    with open(f"{args.outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    label = "conversions" if col == "conversions" else "clicks"
    cost = "CPA" if col == "conversions" else "CPC"
    print(f"\n=== {n} simulated {N_DAYS}-day runs, ${spend:,.0f} spend each, objective: {label} ===")
    print(f"{'strategy':<22}{'total ' + label:>20}{cost:>10}")
    print(f"{'Equal split (baseline)':<22}{summary['baseline_total_mean']:>12.0f} ± {summary['baseline_total_std']:<5.0f}{summary['baseline_cost_per_result']:>10.2f}")
    print(f"{'Ad optimization agent':<22}{summary['agent_total_mean']:>12.0f} ± {summary['agent_total_std']:<5.0f}{summary['agent_cost_per_result']:>10.2f}")
    print(f"{'Oracle (expected)':<22}{summary['oracle_expected_total']:>12.0f}{'':>8}{spend / summary['oracle_expected_total']:>10.2f}")
    print(f"\nAgent vs baseline: {summary['uplift_mean']:+.0f} ± {summary['uplift_ci95']:.0f} {label} "
          f"({summary['uplift_pct']:+.1f}%), better in {summary['win_rate_vs_baseline']:.0%} of runs")
    print(f"After Social's mid-period shift (day {SHIFT_DAY + 1}+): {summary['uplift_pct_after_shift']:+.1f}%")
    print(f"Closes {summary['share_of_oracle_gap_closed_pct']:.0f}% of the baseline-to-oracle gap")
    print(f"\nWrote {args.outdir}/summary.json, per_seed.csv, cumulative_mean.csv; decision log -> {args.log}")


if __name__ == "__main__":
    main()
