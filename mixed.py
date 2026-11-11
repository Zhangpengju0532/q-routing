import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AGENTS = 100
DEMAND = 10.0
EPISODES = 3000
SEEDS = 10
ALPHA = 0.1
EPS_DECAY_EPISODES = 2400
TAIL = 200
ROUTES = [(10.0, 1.0), (14.0, 1.0)]
SHARES = [0, 10, 20, 30, 32, 34, 36, 38, 40, 50, 60, 70, 80, 90, 100]


def run(social_agents, seed):
    rng = random.Random(seed)
    unit = DEMAND / AGENTS
    q = [[0.0, 0.0] for _ in range(AGENTS)]
    tail = []
    for episode in range(EPISODES):
        eps = max(0.0, 1.0 - episode / EPS_DECAY_EPISODES)
        choice = []
        for agent in range(AGENTS):
            if rng.random() < eps or q[agent][0] == q[agent][1]:
                choice.append(rng.randrange(2))
            else:
                choice.append(0 if q[agent][0] > q[agent][1] else 1)
        flow = [unit * choice.count(0), unit * choice.count(1)]
        time = [ROUTES[r][0] + ROUTES[r][1] * flow[r] for r in range(2)]
        social_cost = [time[r] + ROUTES[r][1] * flow[r] for r in range(2)]
        for agent in range(AGENTS):
            r = choice[agent]
            cost = social_cost[r] if agent < social_agents else time[r]
            q[agent][r] += ALPHA * (-cost - q[agent][r])
        if episode >= EPISODES - TAIL:
            social_times = [time[choice[a]] for a in range(social_agents)]
            selfish_times = [time[choice[a]] for a in range(social_agents, AGENTS)]
            tail.append((flow[0], flow[0] * time[0] + flow[1] * time[1],
                         sum(social_times) / len(social_times) if social_times else float("nan"),
                         sum(selfish_times) / len(selfish_times) if selfish_times else float("nan")))
    return [sum(t[i] for t in tail) / len(tail) for i in range(4)]


def theory(share):
    x2 = min(4.0, max(3.0, DEMAND * share / 100))
    x1 = DEMAND - x2
    return x1 * (10 + x1) + x2 * (14 + x2)


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    print("share  flow_1  total_time  time_social  time_selfish")
    for share in SHARES:
        results = [run(AGENTS * share // 100, seed) for seed in range(1, SEEDS + 1)]
        mean = [sum(r[i] for r in results) / SEEDS for i in range(4)]
        rows.append([share] + mean)
        print("%4d%%  %6.2f  %10.2f  %11.2f  %12.2f" % (share, mean[0], mean[1], mean[2], mean[3]))
    with open("results/mixed.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["social_share_percent", "flow_1", "total_time", "time_social", "time_selfish"])
        for row in rows:
            writer.writerow([row[0]] + ["%.3f" % v for v in row[1:]])

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    fine = [s / 10 for s in range(0, 1001)]
    ax.plot(fine, [theory(s) for s in fine], color="0.45", linewidth=1, linestyle=(0, (4, 3)), label="theory", zorder=1)
    ax.plot([r[0] for r in rows], [r[2] for r in rows], color="#2a78d6", linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.2, label="learned (mean of 10 seeds)", zorder=2)
    ax.text(2, 170.12, "user equilibrium 170", color="0.25", fontsize=9)
    ax.text(98, 168.12, "system optimum 168", color="0.25", fontsize=9, ha="right")
    ax.set_xlim(-3, 103)
    ax.set_ylim(167.6, 170.6)
    ax.set_xlabel("share of agents with the social reward (%)")
    ax.set_ylabel("total system travel time")
    ax.set_title("Mixed population: how many social agents does it take?", fontsize=10, loc="left")
    ax.legend(frameon=False, loc="center right", fontsize=9)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig("results/mixed.png", dpi=200)
    print("table written to results/mixed.csv, figure written to results/mixed.png")


main()