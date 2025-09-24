import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load the data from the .pkl file
with open("gradients_data.pkl", "rb") as f:
    data = pickle.load(f)

N = np.array(data["N"])
gradients_HEA = [np.array(g) for g in data["gradients_HEA"]]
gradients_RPA = [np.array(g) for g in data["gradients_RPA"]]

params = [r'$\theta_1$', r'$\theta_2$', r'$\theta_3$',
          r'$\theta_4$', r'$\theta_5$', r'$\theta_6$']

markers = ['o', 's', '^', 'D', 'P', '*']
colors = ['#1f77b4','#2ca02c','#d62728','#9467bd','#ff7f0e','#17becf']

# --- Helper function for fitting, estimating, and plotting ---
def plot_fit_and_approximated_points(x_full, y_full, marker, color, label_prefix):
    """
    Fits on filtered data, plots the fit line, and plots both the original and 
    approximated points with the same style and size.
    """
    y_arr = np.array(y_full)
    x_arr = np.array(x_full)
    marker_size = 80
    
    # --- Create masks for filtering ---
    valid_mask = ~np.isnan(y_arr)
    above_threshold_mask = y_arr >= 1e-27
    fit_mask = valid_mask & above_threshold_mask

    x_fit = x_arr[fit_mask]
    y_fit = y_arr[fit_mask]

    if len(x_fit) < 2:
        print(f"Skipping fit for {label_prefix} due to insufficient data points above threshold.")
        plt.scatter(x_fit, y_fit, marker=marker, color=color, s=marker_size, label=f"{label_prefix} (no fit)")
        return

    # --- Fitting (on filtered data) ---
    log_y = np.log(y_fit)
    slope, intercept, _, _, _ = stats.linregress(x_fit, log_y)
    a = np.exp(intercept)
    b = slope

    # --- Estimate points below threshold ---
    estimate_mask = valid_mask & ~above_threshold_mask
    x_estimate = x_arr[estimate_mask]
    if len(x_estimate) > 0:
        y_estimated = a * np.exp(b * x_estimate)
        # Plot estimated points with the same style and size (no label)
        plt.scatter(x_estimate, y_estimated, marker=marker, color=color, s=marker_size)
    
    # --- Plotting ---
    # Plot the original data points used for the fit (solid markers)
    plt.scatter(x_fit, y_fit, marker=marker, color=color, s=marker_size, label=f"{label_prefix} = {b:.2f}")
    
    # Plot the extended fitted line
    x_line = np.linspace(min(x_full), max(x_full), 200)
    y_line = a * np.exp(b * x_line)
    plt.plot(x_line, y_line, color=color, linestyle='--')


# -----------------------------
# Figure HEA (Figure 6a)
# -----------------------------
plt.figure(figsize=(6,6))
for i in range(6):
    plot_fit_and_approximated_points(N, gradients_HEA[i], markers[i], colors[i], params[i])

plt.yscale('log')
plt.xlabel(r"$N$")
plt.ylabel(r"Var$[\partial_{\theta_k} C]$")
plt.title("HEA Gradients vs. Qubit Number (Approximated)")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figure6_HEA.png", dpi=300)
plt.show()

# -----------------------------
# Figure RPA (Figure 6b)
# -----------------------------
plt.figure(figsize=(6,6))
for i in range(6):
    plot_fit_and_approximated_points(N, gradients_RPA[i], markers[i], colors[i], params[i])

plt.yscale('log')
plt.xlabel(r"$N$")
plt.ylabel(r"Var$[\partial_{\theta_k} C]$")
plt.title("RPA Gradients vs. Qubit Number (Approximated)")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figure6_RPA.png", dpi=300)
plt.show()