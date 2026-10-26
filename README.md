# Ad Optimization Agent

A minimal agent that splits a fixed daily ad budget across **Search, Social and Display** to maximize **conversions** (or clicks), using a simple heuristic plus a daily learning loop. Every budget change is logged with a plain-English reason.

## Run it

```bash
pip install -r requirements.txt     # pandas, numpy, matplotlib

python simulator.py                 # (re)generate data/ad_data.csv  (30 days x 3 channels)
python agent.py --csv data/ad_data.csv   # propose tomorrow's budget from the CSV, print reasons
python evaluate.py                  # closed-loop test vs equal-split baseline (100 seeds)
python plots.py                     # slide charts -> results/*.png
python test_agent.py                # guardrail tests
```

`python agent.py --objective clicks` and `python evaluate.py --objective clicks` optimize clicks instead of conversions.

## Files

| File | Purpose |
|---|---|
| `data/ad_data.csv` | Mock history: `date,channel,spend,impressions,clicks,conversions`, 30 days, equal split |
| `simulator.py` | Generates the CSV and acts as the hidden "market" for evaluation |
| `agent.py` | The agent: metrics, allocation rule, guardrails, decision log |
| `evaluate.py` | Replays 100 simulated 30-day runs: agent vs equal split vs oracle |
| `plots.py` | Charts for the slides |
| `test_agent.py` | Checks the guardrails hold on every day of every run |
| `prompts/system_prompt.md` | Agent role, inputs/outputs, guardrails (for the design doc / an LLM wrapper) |
| `logs/decisions.csv` | Decision log from the demo run: budget before/after, metrics, reason string |
| `results/` | `summary.json`, per-seed results, charts |

## How the agent decides (each day)

1. **Read** all rows dated before today from the CSV / history.
2. **Compute** per-channel CTR, CVR and CPA over the last 7 days.
3. **Score** each channel by conversions per dollar, shrunk toward the blended average so a few lucky days can't swing the budget.
4. **Target** a split proportional to `score^0.5`. The square root is deliberate: ad channels have diminishing returns, so pouring everything into the current best channel wastes money.
5. **Apply guardrails**, then rescale so the total stays exactly at the daily budget:
   - max **±20%** change per channel per day
   - **10% floor** per channel (never 0%, so every channel keeps producing data to learn from)
   - **60% ceiling** per channel
   - fixed total daily budget (acts as the cost cap)
   - first 3 days: equal split (warm-up)
6. **Log** a reason per channel, e.g.
   `Search INCREASE +20.0% ($300.00->$360.00): 7d CPA $3.99 vs blended $7.01, CVR 6.4%, CTR 4.0%; target share 45%; limited to +20% daily step`

## Assumptions

- Total daily budget is **$900** (equal split = $300 per channel), and each channel spends its full budget.
- The data is **mock**. Each channel has a hidden diminishing-returns curve, noise, a weekend effect, and one mid-period shift (Social's conversion rate jumps 55% on day 16). The agent never sees these parameters.
- "Baseline" is the fixed equal split, as suggested in the assignment.
- The evaluation is a **simulation**. Real ad platforms would add attribution delay, auction effects and budget pacing that this does not model.

## Results snapshot (100 simulated runs, $27,000 spend each)

| Strategy | Total conversions | CPA |
|---|---|---|
| Equal split (baseline) | 3,687 ± 72 | $7.32 |
| **Ad optimization agent** | **3,984 ± 80** | **$6.78** |
| Oracle (knows true curves, not achievable) | 4,289 | $6.29 |

- Agent vs baseline: **+297 conversions (+8.0%)**, 95% CI ±18, better in **100 of 100** runs.
- It closes about **47%** of the gap between the baseline and the oracle.
- Charts: `results/cumulative_conversions.png`, `results/budget_over_time.png`.

**Why it helps:** the agent quickly moves money from Display (about $25 CPA) toward Search (about $4.70 CPA) and later toward Social after its conversion rate improves.

**Trade-offs / limitations:**
- The ±20% step cap and 7-day window make it adapt gradually. After Social's shift on day 16 it only moves partway toward Social, which is why it stays below the oracle.
- The floor keeps ~$90+/day in the weakest channel on purpose. That costs some conversions in exchange for continued learning.
- `alpha = 0.5` is a hand-picked damping value; tuning it (or the window length) is the obvious next experiment.
