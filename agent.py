"""
agent.py - Ad Optimization Agent (heuristic explore/exploit budget allocator).

Every day the agent:
  1. READS the performance history (date, channel, spend, impressions, clicks, conversions)
  2. COMPUTES per-channel CTR, CVR and cost-per-result over a trailing window
  3. SCORES each channel by efficiency (results per dollar), shrunk toward the
     blended average so a few lucky days can't swing the budget
  4. PROPOSES a target split (sub-linear in score, because returns diminish)
  5. APPLIES GUARDRAILS: +/-20% max daily change, 10% floor / 60% ceiling per
     channel, fixed total daily budget
  6. LOGS a plain-English reason for every channel's change

Usage (propose tomorrow's budget from an existing CSV):
    python agent.py --csv data/ad_data.csv
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulator import CHANNELS, COLUMNS, TOTAL_DAILY_BUDGET

DECISION_COLUMNS = [
    "date", "channel", "prev_budget", "new_budget", "change_pct", "action",
    "ctr", "cvr", "cost_per_result", "efficiency", "target_share", "reason",
]


@dataclass
class AgentConfig:
    total_budget: float = TOTAL_DAILY_BUDGET
    objective: str = "conversions"     # "conversions" or "clicks"
    window_days: int = 7               # trailing days used to score channels
    warmup_days: int = 3               # hold an equal split until we have this many days
    max_daily_change: float = 0.20     # guardrail: +/-20% of a channel's previous budget per day
    min_share: float = 0.10            # guardrail: floor (keeps exploring; never 0%)
    max_share: float = 0.60            # guardrail: ceiling (limits concentration risk)
    alpha: float = 0.5                 # <1 dampens winner-take-all (diminishing returns)
    prior_days: float = 1.0            # shrinkage strength, in "days of equal-split spend"


def empty_history() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in COLUMNS}).astype(
        {"date": "datetime64[ns]", "channel": "object"})


def load_history(path: str) -> pd.DataFrame:
    """Read the performance CSV and validate its columns."""
    df = pd.read_csv(path, parse_dates=["date"])
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {sorted(missing)}")
    return df[COLUMNS]


def project_to_budget(target: np.ndarray, lo: np.ndarray, hi: np.ndarray, total: float) -> np.ndarray:
    """Find x closest to `target` with lo <= x <= hi and sum(x) == total.

    Solves sum(clip(target + lam, lo, hi)) == total for lam by bisection, which
    spreads any shortfall caused by a binding cap across the other channels.
    """
    a, b = -total, total
    for _ in range(100):
        lam = (a + b) / 2
        if np.clip(target + lam, lo, hi).sum() < total:
            a = lam
        else:
            b = lam
    return np.clip(target + (a + b) / 2, lo, hi)


class BudgetAgent:
    def __init__(self, channels=None, config: AgentConfig | None = None):
        self.channels = list(channels or CHANNELS)
        self.cfg = config or AgentConfig()
        if self.cfg.objective not in ("conversions", "clicks"):
            raise ValueError("objective must be 'conversions' or 'clicks'")

    # ---- step 1-2: performance metrics -------------------------------------------------
    @property
    def _value_col(self) -> str:
        return "conversions" if self.cfg.objective == "conversions" else "clicks"

    @property
    def _cost_label(self) -> str:
        return "CPA" if self.cfg.objective == "conversions" else "CPC"

    def equal_split(self) -> dict:
        return {ch: self.cfg.total_budget / len(self.channels) for ch in self.channels}

    def channel_metrics(self, history: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
        """Trailing-window CTR, CVR, cost-per-result and shrunk efficiency per channel."""
        start = today - pd.Timedelta(days=self.cfg.window_days)
        recent = history[(history["date"] < today) & (history["date"] >= start)]
        g = (recent.groupby("channel")[["spend", "impressions", "clicks", "conversions"]]
             .sum().reindex(self.channels).fillna(0.0))
        g["ctr"] = g["clicks"] / g["impressions"].replace(0, np.nan)
        g["cvr"] = g["conversions"] / g["clicks"].replace(0, np.nan)
        value = g[self._value_col]
        g["cost_per_result"] = g["spend"] / value.replace(0, np.nan)
        # Shrink each channel's results-per-dollar toward the pooled rate so small
        # samples can't dominate: add one "day" of equal-split spend at the pooled rate.
        pooled = value.sum() / max(g["spend"].sum(), 1e-9)
        prior_spend = self.cfg.prior_days * self.cfg.total_budget / len(self.channels)
        g["efficiency"] = (value + pooled * prior_spend) / (g["spend"] + prior_spend)
        return g

    # ---- step 3-6: decide, guard, explain ----------------------------------------------
    def propose(self, history: pd.DataFrame, today: pd.Timestamp, prev_budgets: dict | None = None):
        """Return (budgets, decisions) for `today`, using only data dated before `today`."""
        cfg, chans, total = self.cfg, self.channels, self.cfg.total_budget
        today = pd.Timestamp(today)
        prev = np.array([(prev_budgets or self.equal_split())[c] for c in chans], dtype=float)
        metrics = self.channel_metrics(history, today)
        days_seen = history.loc[history["date"] < today, "date"].nunique()
        floor, ceil = cfg.min_share * total, cfg.max_share * total

        if days_seen < cfg.warmup_days:
            # Not enough data yet: explore with an equal split.
            new = np.array(list(self.equal_split().values()))
            target_share = np.full(len(chans), 1 / len(chans))
            lo = hi = new
            note = f"warm-up (only {days_seen} day(s) of data, need {cfg.warmup_days}); equal split"
        else:
            # Target split: proportional to efficiency^alpha (alpha < 1 respects diminishing returns)
            weights = metrics["efficiency"].to_numpy() ** cfg.alpha
            target = total * weights / weights.sum()
            target_share = target / total
            # Guardrail bounds: daily step cap intersected with floor/ceiling
            lo = np.maximum(floor, prev * (1 - cfg.max_daily_change))
            hi = np.minimum(ceil, prev * (1 + cfg.max_daily_change))
            lo, hi = np.minimum(lo, hi), np.maximum(hi, lo)
            note = ""
            if lo.sum() > total or hi.sum() < total:
                # Previous allocation was outside the guardrails (e.g. external CSV): re-enter via floor/ceiling only
                lo, hi = np.full(len(chans), floor), np.full(len(chans), ceil)
                note = "previous allocation outside guardrails; step cap relaxed once"
            new = project_to_budget(target, lo, hi, total)

        new = np.round(new, 2)
        resid = round(total - new.sum(), 2)           # fix rounding so total is exact
        if resid:
            i = int(np.argmax(hi - new)) if resid > 0 else int(np.argmax(new - lo))
            new[i] = round(new[i] + resid, 2)

        decisions = []
        blended = metrics["spend"].sum() / max(metrics[self._value_col].sum(), 1e-9)
        for i, ch in enumerate(chans):
            decisions.append(self._decision(
                today, ch, prev[i], new[i], lo[i], hi[i], target_share[i] * total,
                metrics.loc[ch], blended, floor, ceil, note, days_seen))
        return dict(zip(chans, new.tolist())), decisions

    def _decision(self, today, ch, prev, new, lo, hi, target, m, blended, floor, ceil, note, days_seen):
        cfg = self.cfg
        change = (new - prev) / prev if prev else 0.0
        action = "hold" if abs(change) < 0.005 else ("increase" if change > 0 else "decrease")
        if note.startswith("warm-up"):
            reason = f"{ch} HOLD ${new:,.2f}: {note}"
        else:
            cpr = f"${m['cost_per_result']:.2f}" if pd.notna(m["cost_per_result"]) else "n/a"
            reason = (f"{ch} {action.upper()} {change:+.1%} (${prev:,.2f}->${new:,.2f}): "
                      f"{cfg.window_days}d {self._cost_label} {cpr} vs blended ${blended:.2f}, "
                      f"CVR {m['cvr']:.1%}, CTR {m['ctr']:.1%}; target share {target / cfg.total_budget:.0%}")
            if abs(target - new) > 1.0:                 # a guardrail is actually binding
                at_hi, at_lo = abs(new - hi) <= 0.01, abs(new - lo) <= 0.01
                if at_hi and hi >= ceil - 0.01:
                    reason += f"; held at {cfg.max_share:.0%} share ceiling"
                elif at_hi:
                    reason += f"; limited to +{cfg.max_daily_change:.0%} daily step"
                elif at_lo and lo <= floor + 0.01:
                    reason += f"; held at {cfg.min_share:.0%} floor (keeps exploring)"
                elif at_lo:
                    reason += f"; limited to -{cfg.max_daily_change:.0%} daily step"
                else:
                    reason += "; total budget is fixed, so the other channels' step caps limit how far it can move"
            if note:
                reason += f"; {note}"
        return {
            "date": today.strftime("%Y-%m-%d"), "channel": ch,
            "prev_budget": round(float(prev), 2), "new_budget": round(float(new), 2),
            "change_pct": round(float(change) * 100, 2), "action": action,
            "ctr": None if pd.isna(m["ctr"]) else round(float(m["ctr"]), 4),
            "cvr": None if pd.isna(m["cvr"]) else round(float(m["cvr"]), 4),
            "cost_per_result": None if pd.isna(m["cost_per_result"]) else round(float(m["cost_per_result"]), 2),
            "efficiency": round(float(m["efficiency"]), 4),
            "target_share": round(float(target / cfg.total_budget), 4), "reason": reason,
        }


def write_decision_log(decisions: list[dict], path: str, append: bool = True) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(decisions, columns=DECISION_COLUMNS)
    df.to_csv(path, mode="a" if append and os.path.exists(path) else "w",
              header=not (append and os.path.exists(path)), index=False)


def main():
    p = argparse.ArgumentParser(description="Propose tomorrow's channel budgets from a performance CSV.")
    p.add_argument("--csv", default="data/ad_data.csv")
    p.add_argument("--budget", type=float, default=TOTAL_DAILY_BUDGET, help="total daily budget ($)")
    p.add_argument("--objective", choices=["conversions", "clicks"], default="conversions")
    p.add_argument("--log", default="logs/decisions.csv")
    args = p.parse_args()

    history = load_history(args.csv)
    agent = BudgetAgent(config=AgentConfig(total_budget=args.budget, objective=args.objective))
    today = history["date"].max() + pd.Timedelta(days=1)
    last = history[history["date"] == history["date"].max()].set_index("channel")["spend"]
    prev = {ch: float(last.get(ch, 0.0)) for ch in agent.channels}   # yesterday's actual spend

    print(f"\nLast 7 days of performance (as of {today.date()}):")
    print(agent.channel_metrics(history, today)[["spend", "impressions", "clicks", "conversions",
                                                 "ctr", "cvr", "cost_per_result"]].round(4).to_string())
    budgets, decisions = agent.propose(history, today, prev)
    print(f"\nProposed budgets for {today.date()} (total ${sum(budgets.values()):,.2f}):")
    for d in decisions:
        print("  " + d["reason"])
    write_decision_log(decisions, args.log)
    print(f"\nDecisions appended to {args.log}")


if __name__ == "__main__":
    main()
