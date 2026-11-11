import csv
import os
import random

AGENTS = 100
DEMAND = 10.0
EPISODES = 3000
SEEDS = 10
ALPHA = 0.1
EPS_START = 1.0
EPS_END = 0.0
EPS_DECAY_EPISODES = 2400
TAIL = 200
ROUTES = [(10.0, 1.0), (14.0, 1.0)]


def travel_time(route, x):
    a, b = ROUTES[route]
    return a + b * x


def marginal_external_cost(route, x):
    a, b = ROUTES[route]
    return b * x


def run(mode, seed):
    rng = random.Random(seed)
    unit = DEMAND / AGENTS
    q = [[0.0, 0.0] for _ in range(AGENTS)]
    history = []
    for episode in range(EPISODES):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * episode / EPS_DECAY_EPISODES)
        choice = []
        for agent in range(AGENTS):
            if rng.random() < eps or q[agent][0] == q[agent][1]:
                choice.append(rng.randrange(2))
            else:
                choice.append(0 if q[agent][0] > q[agent][1] else 1)
        flow = [unit * choice.count(0), unit * choice.count(1)]
        time = [travel_time(r, flow[r]) for r in range(2)]
        cost = list(time)
        if mode == "social":
            cost = [time[r] + marginal_external_cost(r, flow[r]) for r in range(2)]
        for agent in range(AGENTS):
            r = choice[agent]
            q[agent][r] += ALPHA * (-cost[r] - q[agent][r])
        history.append((flow[0], flow[1], flow[0] * time[0] + flow[1] * time[1]))
    return history


def main():
    os.makedirs("results", exist_ok=True)
    rows = []
    for mode in ("selfish", "social"):
        finals = []
        for seed in range(1, SEEDS + 1):
            history = run(mode, seed)
            tail = history[-TAIL:]
            x1 = sum(h[0] for h in tail) / TAIL
            total = sum(h[2] for h in tail) / TAIL
            finals.append((x1, total))
            for episode, h in enumerate(history):
                rows.append((mode, seed, episode, "%.2f" % h[0], "%.2f" % h[1], "%.3f" % h[2]))
            print(mode, "seed", seed, "route 1 flow", "%.2f" % x1, "total time", "%.2f" % total)
        mean_x1 = sum(f[0] for f in finals) / SEEDS
        mean_total = sum(f[1] for f in finals) / SEEDS
        print(mode.upper(), "mean over", SEEDS, "seeds: route 1 flow", "%.2f" % mean_x1, "| route 2 flow", "%.2f" % (DEMAND - mean_x1), "| total time", "%.2f" % mean_total)
    with open("results/history.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "seed", "episode", "flow_1", "flow_2", "total_time"])
        writer.writerows(rows)
    print("theory: user equilibrium 7.00 / 3.00 / 170.00, system optimum 6.00 / 4.00 / 168.00")
    print("history written to results/history.csv")


main()