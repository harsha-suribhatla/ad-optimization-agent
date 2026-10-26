"""plots.py - Make the two slide charts from the files evaluate.py wrote.

    python plots.py   ->  results/cumulative_conversions.png, results/budget_over_time.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from simulator import CHANNELS, SHIFT_DAY

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9"
SERIES = {"Search": "#2a78d6", "Social": "#eb6834", "Display": "#1baf7a"}   # fixed slot order


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xlabel("Day", color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, length=0)


def cumulative(path="results"):
    d = pd.read_csv(f"{path}/cumulative_mean.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    x = d["day"] + 1
    ax.plot(x, d["oracle_expected"], color=MUTED, linewidth=1.8, linestyle=":", label="Oracle (knows true curves)")
    ax.plot(x, d["baseline"], color="#898781", linewidth=2, label="Equal split (baseline)")
    ax.plot(x, d["agent"], color=SERIES["Search"], linewidth=2.5, label="Ad optimization agent")
    ax.axvline(SHIFT_DAY + 1, color=GRID, linewidth=1.2)
    ax.text(SHIFT_DAY + 1.3, d["oracle_expected"].max() * 0.04, "Social CVR jumps", color=MUTED, fontsize=9)
    style(ax, "Cumulative conversions, mean of 100 simulated runs", "Conversions")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{path}/cumulative_conversions.png", dpi=200, facecolor=SURFACE)


def budgets(path="results"):
    d = pd.read_csv(f"{path}/demo_budgets_agent.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    x = d["day"] + 1
    for ch in CHANNELS:
        ax.plot(x, d[ch], color=SERIES[ch], linewidth=2.5, label=ch)
        ax.text(x.iloc[-1] + 0.4, d[ch].iloc[-1], f"{ch} ${d[ch].iloc[-1]:.0f}", color=INK, fontsize=9, va="center")
    ax.axvline(SHIFT_DAY + 1, color=GRID, linewidth=1.2)
    ax.text(SHIFT_DAY + 1.3, 60, "Social CVR jumps", color=MUTED, fontsize=9)
    ax.set_xlim(1, 36)
    ax.set_ylim(0, 480)
    style(ax, "Agent's daily budget by channel (seed 42)", "Daily budget ($)")
    ax.legend(frameon=False, labelcolor=INK, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(f"{path}/budget_over_time.png", dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    cumulative()
    budgets()
    print("Wrote results/cumulative_conversions.png and results/budget_over_time.png")
