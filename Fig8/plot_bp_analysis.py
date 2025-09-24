# plot_bp_results.py

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pickle
import os

# Define the filename where results are stored
RESULTS_FILENAME = "bp_analysis_results.pkl"

def create_fit_plot(ga_results, random_results, title, save_filename):
    """
    Generates a plot showing the variance of the gradient vs. N and a fit line.
    This function is copied directly from the original script.
    """
    print(f"\n📊 Generating plot: '{title}'...")

    plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(10, 8))

    # --- Plotting GA Results ---
    ga_qubits = np.array(ga_results['qubits'])
    ga_variances = np.array(ga_results['variances'])

    ax.scatter(ga_qubits, ga_variances, color='blue', marker='o', s=100, zorder=3, label='GA Optimized')

    # Exponential Fit for GA
    valid_indices_ga = ga_variances > 1e-12
    if np.sum(valid_indices_ga) > 1:
        slope, intercept, _, _, _ = stats.linregress(ga_qubits[valid_indices_ga], np.log(ga_variances[valid_indices_ga]))
        fit_line_ga = np.exp(intercept) * np.exp(slope * ga_qubits)
        ax.plot(ga_qubits, fit_line_ga, color='blue', linestyle='--', linewidth=2, zorder=2, label=f'GA Fit (Slope: {slope:.2f})')

    # --- Plotting Random Circuits Results ---
    random_qubits = np.array(random_results['qubits'])
    random_variances = np.array(random_results['variances'])

    ax.scatter(random_qubits, random_variances, color='red', marker='s', s=100, zorder=3, label='Random Circuit')

    # Exponential Fit for Random
    valid_indices_random = random_variances > 1e-12
    if np.sum(valid_indices_random) > 1:
        slope, intercept, _, _, _ = stats.linregress(random_qubits[valid_indices_random], np.log(random_variances[valid_indices_random]))
        fit_line_random = np.exp(intercept) * np.exp(slope * random_qubits)
        ax.plot(random_qubits, fit_line_random, color='red', linestyle='--', linewidth=2, zorder=2, label=f'Random Fit (Slope: {slope:.2f})')

    ax.set_yscale('log')
    ax.set_xlabel("Number of Qubits (N)", fontsize=14)
    ax.set_ylabel(r"Var($\partial C / \partial \theta_1$)", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_filename, dpi=300)
    print(f"✅ Plot saved to {save_filename}")
    plt.show()

if __name__ == "__main__":
    # Check if the data file exists
    if not os.path.exists(RESULTS_FILENAME):
        print(f"Error: Results file '{RESULTS_FILENAME}' not found.")
        print("Please run 'run_bp_analysis.py' first to generate the data.")
    else:
        print(f"Loading data from '{RESULTS_FILENAME}'...")
        # Load the data from the pickle file
        with open(RESULTS_FILENAME, 'rb') as f:
            loaded_data = pickle.load(f)

        # Extract the results for each category
        ga_results = loaded_data["ga_optimized"]
        random_results = loaded_data["random"]

        # Call the plotting function
        create_fit_plot(ga_results, random_results, 
                        "Barren Plateau Analysis: GA-Optimized vs. Random Circuits", 
                        "bp_analysis_variance_vs_n.png")