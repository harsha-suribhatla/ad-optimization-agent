"""
simulator.py - Mock ad market for the Ad Optimization Agent.

This module plays two roles:
  1. It generates the mock CSV (14-30 days x 3 channels) that the agent reads.
  2. It acts as the "environment" in evaluate.py: given a daily budget split,
     it returns the clicks/conversions that split would have produced, so the
     agent can be scored against baselines in a closed loop.

Each channel has a hidden response curve with DIMINISHING RETURNS (the first
dollars buy more impressions than the last dollars), plus day-to-day noise,
a weekend effect, and one mid-period shift (Social's conversion rate jumps on
day 15) so the agent has something to adapt to.

The agent never sees these parameters - it only sees the CSV columns:
    date, channel, spend, impressions, clicks, conversions
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

CHANNELS = ["Search", "Social", "Display"]
TOTAL_DAILY_BUDGET = 900.0            # dollars per day, split across channels
START_DATE = pd.Timestamp("2026-09-25")
N_DAYS = 30
SHIFT_DAY = 15                        # day index when Social's CVR jumps
COLUMNS = ["date", "channel", "spend", "impressions", "clicks", "conversions"]


@dataclass(frozen=True)
class ChannelModel:
    """Hidden ground truth for one channel."""
    name: str
    max_impressions: float   # impressions/day reachable if spend were unlimited
    saturation: float        # dollar scale at which returns flatten out
    ctr: float               # click-through rate (clicks / impressions)
    cvr: float               # conversion rate (conversions / clicks)

    def impressions(self, spend):
        """Expected impressions for a given spend (concave: diminishing returns)."""
        return self.max_impressions * (1.0 - np.exp(-np.asarray(spend, dtype=float) / self.saturation))


TRUE_MODELS = {
    # Search: high intent, small audience -> saturates quickly
    "Search": ChannelModel("Search", max_impressions=60_000, saturation=450, ctr=0.040, cvr=0.060),
    # Social: big audience, medium intent (CVR improves mid-period, see multipliers)
    "Social": ChannelModel("Social", max_impressions=150_000, saturation=700, ctr=0.018, cvr=0.035),
    # Display: huge audience, low intent -> weakest per dollar
    "Display": ChannelModel("Display", max_impressions=400_000, saturation=900, ctr=0.005, cvr=0.020),
}


def multipliers(channel: str, day_idx: int, date: pd.Timestamp) -> tuple[float, float]:
    """Time-varying (ctr_multiplier, cvr_multiplier) for a channel on a given day."""
    ctr_m, cvr_m = 1.0, 1.0
    if date.weekday() >= 5:                       # weekend effects
        if channel == "Search":
            cvr_m *= 0.90
        if channel == "Social":
            ctr_m *= 1.10
    if channel == "Social" and day_idx >= SHIFT_DAY:   # mid-period shift (e.g. a promo lands)
        cvr_m *= 1.55
    return ctr_m, cvr_m


def expected_conversions(channel: str, spend, day_idx: int, date: pd.Timestamp, models=TRUE_MODELS):
    """Noise-free expected conversions for a channel at a given spend (vectorised over spend)."""
    m = models[channel]
    ctr_m, cvr_m = multipliers(channel, day_idx, date)
    return m.impressions(spend) * (m.ctr * ctr_m) * (m.cvr * cvr_m)


class AdEnvironment:
    """Closed-loop environment: budgets in, observed performance rows out."""

    def __init__(self, seed: int = 0, models=TRUE_MODELS):
        self.rng = np.random.default_rng(seed)
        self.models = models

    def step(self, date: pd.Timestamp, day_idx: int, budgets: dict) -> pd.DataFrame:
        rows = []
        for ch in CHANNELS:
            m = self.models[ch]
            spend = float(budgets[ch])
            ctr_m, cvr_m = multipliers(ch, day_idx, date)
            # day-level noise on the rates, then sampling noise on the counts
            ctr = min(m.ctr * ctr_m * self.rng.lognormal(0, 0.05), 1.0)
            cvr = min(m.cvr * cvr_m * self.rng.lognormal(0, 0.08), 1.0)
            impressions = int(self.rng.poisson(m.impressions(spend)))
            clicks = int(self.rng.binomial(impressions, ctr))
            conversions = int(self.rng.binomial(clicks, cvr))
            rows.append({
                "date": date, "channel": ch, "spend": round(spend, 2),
                "impressions": impressions, "clicks": clicks, "conversions": conversions,
            })
        return pd.DataFrame(rows, columns=COLUMNS)


def generate_baseline_csv(path: str, seed: int = 42, days: int = N_DAYS) -> pd.DataFrame:
    """Write a `days`-day CSV produced by an equal-split budget (the 'historical' log)."""
    env = AdEnvironment(seed)
    equal = {ch: TOTAL_DAILY_BUDGET / len(CHANNELS) for ch in CHANNELS}
    frames = []
    for d in range(days):
        date = START_DATE + pd.Timedelta(days=d)
        frames.append(env.step(date, d, equal))
    df = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = df.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the mock ad performance CSV.")
    parser.add_argument("--out", default="data/ad_data.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", type=int, default=N_DAYS)
    args = parser.parse_args()
    df = generate_baseline_csv(args.out, args.seed, args.days)
    print(f"Wrote {len(df)} rows ({args.days} days x {len(CHANNELS)} channels) to {args.out}")
