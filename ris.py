# Both the codes commented one and thereafter are correct, the second one is made to run faster.

# Full RIS with KPT, Theta, and Optimized Seed Selection (Parallelized + Numba)

import os
import math
import time
import random
import numpy as np
import pandas as pd
import networkx as nx
import ast
from numba import njit
from joblib import Parallel, delayed
from collections import defaultdict, Counter
from threading import Thread
import multiprocessing as mp

mp.set_start_method("spawn", force=True)

# Configurations
NUM_CPUS = 24
SIMULATIONS = 10000
BUDGETS = [10, 20, 30, 40, 50]
EPSILON = 0.3
L = 1
GRAPH_VERSIONS = {
    "trivalency": "euemail_trivalency.txt",
    "uniform": "euemail_uniform.txt",
    "weighted": "euemail_weighted.txt"
}
LOG_FILE = "live_log.csv"

# Load cost-benefit data
def load_data():
    with open('cost.txt', 'r') as f:
        costs = ast.literal_eval(f.read().strip())
    with open('benefit.txt', 'r') as f:
        benefits = ast.literal_eval(f.read().strip())
    return {int(k): float(v) for k, v in costs.items()}, {int(k): float(v) for k, v in benefits.items()}

def to_matrix(G):
    n = max(G.nodes()) + 1
    adj = np.zeros((n, n), dtype=np.int32)
    prob = np.zeros((n, n), dtype=np.float64)
    for u, v, d in G.edges(data=True):
        adj[u][v] = 1
        prob[u][v] = d.get('weight', 0.1)
    return adj, prob

@njit
def simulate_diffusion(seed_set, adj_matrix, prob_matrix):
    n = adj_matrix.shape[0]
    active = np.zeros(n, dtype=np.bool_)
    newly_active = np.zeros(n, dtype=np.bool_)
    for seed in seed_set:
        active[seed] = True
        newly_active[seed] = True
    steps = 0
    while np.any(newly_active):
        next_active = np.zeros(n, dtype=np.bool_)
        for u in range(n):
            if newly_active[u]:
                for v in range(n):
                    rand_val = round(np.random.rand(), 3)
                    if adj_matrix[u][v] and not active[v] and rand_val < prob_matrix[u][v]:
                        next_active[v] = True
        active |= next_active
        newly_active = next_active
        steps += 1
    return np.where(active)[0], steps, np.sum(active)

def simulate_parallel(seed_set, adj_matrix, prob_matrix, num_simulations):
    return Parallel(n_jobs=NUM_CPUS, backend="threading")(
        delayed(simulate_diffusion)(seed_set, adj_matrix, prob_matrix)
        for _ in range(num_simulations)
    )

@njit
def generate_rr_set(start_node, adj_matrix, prob_matrix):
    n = adj_matrix.shape[0]
    rr_set = np.zeros(n, dtype=np.bool_)
    queue = [start_node]
    rr_set[start_node] = True
    while queue:
        current = queue.pop()
        for pred in range(n):
            if adj_matrix[pred][current] and not rr_set[pred]:
                rand_val = round(np.random.rand(), 3)
                if rand_val < prob_matrix[pred][current]:
                    rr_set[pred] = True
                    queue.append(pred)
    return rr_set

def batch_generate_rr_sets(batch, adj_matrix, prob_matrix):
    return [generate_rr_set(node, adj_matrix, prob_matrix) for node in batch]

def generate_rr_sets(start_nodes, adj_matrix, prob_matrix, batch_size=10):
    batches = [start_nodes[i:i+batch_size] for i in range(0, len(start_nodes), batch_size)]
    all_rr_sets = Parallel(n_jobs=NUM_CPUS, backend="threading")(
        delayed(batch_generate_rr_sets)(batch, adj_matrix, prob_matrix)
        for batch in batches
    )
    return [rr for batch in all_rr_sets for rr in batch]

def async_log_writer(log_rows, columns):
    def write():
        df = pd.DataFrame(log_rows, columns=columns)
        if not os.path.exists(LOG_FILE):
            df.to_csv(LOG_FILE, index=False)
        else:
            df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    Thread(target=write).start()

def log_binomial(n, k):
    if k < 0 or k > n:
        return -float('inf')
    if k == 0 or k == n:
        return 0
    log_result = 0
    for i in range(1, k + 1):
        log_result += math.log(n - i + 1) - math.log(i)
    return log_result

def kpt_estimation(G, k, cost_dict, benefit_dict, adj, prob):
    n = max(G.nodes()) + 1
    m = G.number_of_edges()
    eta = n
    ratios = np.array([benefit_dict.get(v, 0.0) / max(cost_dict.get(v, 1e-5), 1e-5) for v in range(n)])
    probs = ratios / ratios.sum() if ratios.sum() > 0 else np.full(n, 1/n)

    for i in range(1, int(math.log2(eta)) + 1):
        c_i = int((6 * L * math.log(eta) + 6 * math.log(math.log2(eta))) * (2 ** i))
        sum_profit = 0
        for _ in range(c_i):
            node = np.random.choice(n, p=probs)
            rr_set = generate_rr_set(node, adj, prob)
            wr = sum(adj[:, j].sum() for j in np.where(rr_set)[0])
            kappa = 1 - (1 - (wr / m)) ** k if m > 0 else 0
            sum_profit += kappa
        if sum_profit / c_i > 1 / (2 ** i):
            return max((eta * sum_profit) / (2 * c_i), 1.0)
    return 1.0

def greedy_node_selection(rr_sets, cost_dict, benefit_dict, budget):
    n = len(rr_sets[0])
    covered = np.zeros(len(rr_sets), dtype=np.bool_)
    seed_set = []
    remaining_budget = budget

    while True:
        score = np.zeros(n)
        for i, rr in enumerate(rr_sets):
            if not covered[i]:
                for node in np.where(rr)[0]:
                    score[node] += 1
        ratios = [(i, score[i] / cost_dict.get(i, 1e9)) for i in range(n) if i not in seed_set and cost_dict.get(i, 1e9) <= remaining_budget]
        if not ratios:
            break
        selected = max(ratios, key=lambda x: x[1])[0]
        seed_set.append(selected)
        remaining_budget -= cost_dict.get(selected, 0)
        for i, rr in enumerate(rr_sets):
            if rr[selected]:
                covered[i] = True
    return seed_set

def ris_algorithm(G, budget, cost_dict, benefit_dict):
    global last_kpt, last_theta
    n = max(G.nodes()) + 1
    min_cost = min([v for v in cost_dict.values() if v > 0])
    k = max(1, int(budget / min_cost))
    adj, prob = to_matrix(G)
    kpt = kpt_estimation(G, k, cost_dict, benefit_dict, adj, prob)
    last_kpt = kpt
    ln_binom = log_binomial(n, k)
    theta = int(((8 + 2 * EPSILON) * n * (L * math.log(n) + ln_binom + math.log(2))) / (kpt * EPSILON ** 2))
    last_theta = theta
    print(f"Computed theta: {theta} RR sets")
    nodes = list(G.nodes())
    ratios = np.array([benefit_dict.get(v, 0.0) / max(cost_dict.get(v, 1e-5), 1e-5) for v in range(n)])
    probs = ratios / ratios.sum() if ratios.sum() > 0 else np.full(n, 1/n)
    start_nodes = np.random.choice(n, size=theta, p=probs)
    rr_sets = generate_rr_sets(start_nodes, adj, prob, batch_size=20)
    seed_set = greedy_node_selection(rr_sets, cost_dict, benefit_dict, budget)
    return seed_set, sum([cost_dict.get(i, 0) for i in seed_set])

def run_ris():
    from tqdm import tqdm
    costs, benefits = load_data()
    results = []
    total_tasks = len(GRAPH_VERSIONS) * len(BUDGETS)
    pbar = tqdm(total=total_tasks, desc="Processing", unit="task")

    # Load previous results for crash recovery
    completed = set()
    if os.path.exists(LOG_FILE):
        try:
            prev_df = pd.read_csv(LOG_FILE)
            for _, row in prev_df.iterrows():
                completed.add((row['Model'], int(row['Budget'])))
        except Exception as e:
            print(f"⚠️ Error reading previous log file: {e}")

    for name, file in GRAPH_VERSIONS.items():
        G = nx.read_weighted_edgelist(file, create_using=nx.DiGraph(), nodetype=int)
        adj_matrix, prob_matrix = to_matrix(G)

        for budget in BUDGETS:
            if (name, budget) in completed:
                print(f"⏩ Skipping completed: {name}, Budget: {budget}")
                pbar.update(1)
                continue

            start_time_budget = time.time()
            seed_set, seed_cost = ris_algorithm(G, budget, costs, benefits)
            sims = simulate_parallel(np.array(seed_set, dtype=np.int32), adj_matrix, prob_matrix, SIMULATIONS)
            avg_benefit = np.mean([sum(benefits.get(i, 0) for i in s[0]) for s in sims])
            avg_steps = np.mean([s[1] for s in sims])
            all_activated_nodes = [list(s[0]) for s in sims]

            motif_start_time = time.time()
            # Placeholder: motif_exec_time will be updated later
            result = {
                "KPT": round(last_kpt, 4),
                "Theta": last_theta,
                "Model": name,
                "Budget": budget,
                "Seed_Set": str(seed_set),
                "Seed_Size": len(seed_set),
                "Seed_Cost": float(seed_cost),
                "Remaining_Budget": budget - seed_cost,
                "Avg_Benefit": float(avg_benefit),
                "Profit": float(avg_benefit - seed_cost),
                "Avg_Timestep": round(avg_steps),
                "RIS_Execution_Time": round(time.time() - start_time_budget, 2),
                "Motif_Execution_Time": 0.0,
                "Total_Execution_Time": 0.0,
                "Activated_Nodes": all_activated_nodes,
                "Simulation_Results": sims,
                "Benefits": benefits,
                "Threshold": 0
            }

            results.append(result)
            print(f"✅ {name} | Budget: {budget} | Profit: {avg_benefit - seed_cost:.2f} | Steps: {avg_steps:.2f} | KPT: {last_kpt:.4f} | Theta: {last_theta} | RIS Time: {result['RIS_Execution_Time']:.2f}s")
            pbar.update(1)

    pbar.close()

    async_log_writer(results, ["KPT", "Theta", "Activated_Nodes", "Motif_Execution_Time", "RIS_Execution_Time", "Total_Execution_Time", 
                               "Model", "Budget", "Seed_Set", "Seed_Size", "Seed_Cost", "Remaining_Budget", "Avg_Benefit", "Profit", "Avg_Timestep", "Threshold"])


    for model in GRAPH_VERSIONS:
        model_df = pd.DataFrame([r for r in results if r['Model'] == model])
        model_df = model_df[["Model", "Budget", "KPT", "Theta", "Motif_Execution_Time", "RIS_Execution_Time", "Total_Execution_Time", "Activated_Nodes", "Seed_Set", "Seed_Size", "Seed_Cost", "Remaining_Budget", "Avg_Benefit", "Profit", "Avg_Timestep", "Threshold"]]
        if not model_df.empty:
            model_df.to_csv(f"RIS_Results_{model}.csv", index=False)

    return results

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     start_time_all = time.time()
#     final_results = run_ris()
#     pd.DataFrame(final_results).to_excel("RIS_Results.xlsx", index=False)
#     print("✅ All results saved to RIS_Results.xlsx")
#     print(f"⏱️ Total Execution Time: {round(time.time() - start_time_all, 2)} seconds")

#     # Plotting KPT vs Budget and Theta vs Budget per model
#     df = pd.DataFrame(final_results)
#     for model in df['Model'].unique():
#         subset = df[df['Model'] == model]

#         plt.figure()
#         plt.plot(subset['Budget'], subset['KPT'], marker='o')
#         plt.title(f"KPT vs Budget ({model})")
#         plt.xlabel("Budget")
#         plt.ylabel("KPT")
#         plt.grid(True)
#         plt.savefig(f"KPT_vs_Budget_{model}.png")

#         plt.figure()
#         plt.plot(subset['Budget'], subset['Theta'], marker='s')
#         plt.title(f"Theta vs Budget ({model})")
#         plt.xlabel("Budget")
#         plt.ylabel("Theta")
#         plt.grid(True)
#         plt.savefig(f"Theta_vs_Budget_{model}.png")