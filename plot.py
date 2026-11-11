import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WINDOW = 25
COLORS = {"selfish": "#2a78d6", "social": "#eb6834"}
LABELS = {"selfish": "selfish reward: travel time", "social": "social reward: travel time + MEC"}
THEORY = {"selfish": (7.0, "user equilibrium 7.0"), "social": (6.0, "system optimum 6.0")}

data = {}
with open("results/history.csv") as fh:
    for row in csv.DictReader(fh):
        data.setdefault(row["mode"], {}).setdefault(int(row["seed"]), []).append(float(row["flow_1"]))


def smooth(series):
    out = []
    total = 0.0
    for i, value in enumerate(series):
        total += value
        if i >= WINDOW:
            total -= series[i - WINDOW]
        out.append(total / min(i + 1, WINDOW))
    return out


fig, ax = plt.subplots(figsize=(7.0, 4.2))
for mode in ("selfish", "social"):
    runs = [smooth(series) for series in data[mode].values()]
    episodes = range(len(runs[0]))
    low = [min(run[i] for run in runs) for i in episodes]
    high = [max(run[i] for run in runs) for i in episodes]
    mean = [sum(run[i] for run in runs) / len(runs) for i in episodes]
    ax.fill_between(episodes, low, high, color=COLORS[mode], alpha=0.18, linewidth=0)
    ax.plot(episodes, mean, color=COLORS[mode], linewidth=2, label=LABELS[mode])
    level, text = THEORY[mode]
    ax.axhline(level, color="0.45", linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.text(len(mean) * 1.01, level, text, color="0.25", fontsize=9, va="center")
ax.axvline(2400, color="0.8", linewidth=1)
ax.text(2400, 4.72, "exploration ends ", color="0.35", fontsize=9, ha="right")
ax.set_xlim(0, len(mean))
ax.set_ylim(4.6, 7.6)
ax.set_xlabel("episode (day)")
ax.set_ylabel("flow on route 1")
ax.set_title("100 Q-learning agents, two routes, 10 seeds (line: mean, band: min to max)", fontsize=10, loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.grid(axis="y", color="0.9", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig("results/learning_curve.png", dpi=200, bbox_inches="tight")
print("figure written to results/learning_curve.png")