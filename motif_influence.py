import ast
import numpy as np
import time
from joblib import Parallel, delayed

# Config
NUM_CPUS = 24

def load_motifs(motif_file):
    motifs = []
    try:
        with open(motif_file, "r") as f:
            for line in f:
                motif = set(ast.literal_eval(line.strip()))
                motifs.append(motif)
    except FileNotFoundError:
        print(f"Error: {motif_file} not found.")
        return []
    except Exception as e:
        print(f"Error parsing {motif_file}: {e}")
        return []
    return motifs

def compute_motif_influence(activated_nodes, motifs, benefits, threshold):
    motif_influence_simulation = set()
    activated_set = set(activated_nodes)
    for motif in motifs:
        common_nodes = motif & activated_set
        if len(common_nodes) >= threshold:
            motif_influence_simulation.update(motif)
    return motif_influence_simulation

def compute_motif_profit(activated_nodes, motifs, benefits, threshold, seed_cost):
    influenced_nodes = compute_motif_influence(activated_nodes, motifs, benefits, threshold)
    motif_profit = sum(benefits.get(i, 0) for i in influenced_nodes) - seed_cost
    return motif_profit

def process_motif_profits(algorithm_results, motif_file, threshold):
    # threshold = threshold
    motifs = load_motifs(motif_file)
    final_results = []
    for result in algorithm_results:
        new_result = result.copy()
        if not motifs:
            print("No motifs loaded, adding default motif metrics.")
            new_result.update({
                "Avg_Motif_Profit": 0.0,
                "Max_Motif_Profit": 0.0,
                "Motif_Profs_List": "[]",
                "Motif_Execution_Time": 0.0,
                "Threshold": threshold
            })
            new_result.pop("Simulation_Results", None)
            new_result.pop("Benefits", None)
            final_results.append(new_result)
            continue
        
        start_time = time.time()
        sims = result["Simulation_Results"]
        benefits = result["Benefits"]
        seed_cost = result["Seed_Cost"]
        motif_profits = Parallel(n_jobs=NUM_CPUS)(
            delayed(compute_motif_profit)(
                np.where(sim_result[0])[0], motifs, benefits, threshold, seed_cost
            ) for sim_result in sims
        )
        avg_motif_profit = np.mean(motif_profits) if motif_profits else 0
        max_motif_profit = np.max(motif_profits) if motif_profits else 0
        motif_execution_time = time.time() - start_time
        total_time = new_result.get("RIS_Execution_Time", 0.0) + motif_execution_time

        new_result.update({
            "Avg_Motif_Profit": avg_motif_profit,
            "Max_Motif_Profit": max_motif_profit,
            "Motif_Profs_List": str(motif_profits),
            "Motif_Execution_Time": round(motif_execution_time, 2),
            "Total_Execution_Time": round(total_time, 2),
            "Threshold": threshold
        })

        new_result.pop("Simulation_Results", None)
        new_result.pop("Benefits", None)
        final_results.append(new_result)
    
    return final_results


