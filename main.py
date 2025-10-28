import pandas as pd
import time
import os
from ris import run_ris
from motif_influence import process_motif_profits
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Config
FINAL_RESULT_FILE = "RIS_Results.xlsx"
MOTIF_FILE = "motifs_size2.txt"
THRESHOLDS = [1, 2]

def main():
    total_start_time = time.time()

    if os.path.exists(FINAL_RESULT_FILE):
        print(f"Reusing existing {FINAL_RESULT_FILE}")
        algorithm_results = pd.read_excel(FINAL_RESULT_FILE).to_dict(orient="records")
    else:
        print("Running RIS algorithm...")
        algorithm_results = run_ris()
        pd.DataFrame(algorithm_results).to_excel(FINAL_RESULT_FILE, index=False)
        print(f"All results saved to {FINAL_RESULT_FILE}")
        print(f"Total Execution Time: {round(time.time() - total_start_time, 2)} seconds")

    # Process motif profits for both thresholds
    threshold_results = {}
    for threshold in THRESHOLDS:
        print(f"Processing motif profits for threshold = {threshold}")
        results_with_motif = process_motif_profits(algorithm_results, motif_file=MOTIF_FILE, threshold=threshold)
        threshold_results[threshold] = results_with_motif
        print(f"Processed motif results for threshold = {threshold}")

    with pd.ExcelWriter("RIS_Motif_Results_DualThreshold.xlsx") as writer:
        for threshold, results in threshold_results.items():
            pd.DataFrame(results).to_excel(writer, sheet_name=f"Threshold_{threshold}", index=False)

    for threshold, results in threshold_results.items():
        for result in results:
            print(f"Budget {result['Budget']}, Threshold {threshold}: Motif Execution Time = {result.get('Motif_Execution_Time', 0.0)} seconds")

    total_execution_time = time.time() - total_start_time
    print(f"Final motif results saved to RIS_Motif_Results_DualThreshold.xlsx")
    print(f"Total Execution Time (all budgets): {round(total_execution_time, 2)} seconds")

if __name__ == "__main__":
    main()
