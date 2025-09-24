import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- Configuration (should match the generation script) ---
NUM_QUBITS = 2

# --- Plotting Functions ---

def plot_binned_average(ax, x_data, y_data, bin_size=2, poly_degree=3, label=None, marker_color='yellow', line_color='red', fix_first_point=True):
    """
    Calculates and plots a smooth polynomial trendline for data on a given axis.
    The 'poly_degree' parameter controls the curvature.
    - poly_degree=1: Straight line
    - poly_degree=2: Simple curve (quadratic)
    - poly_degree=3: More complex curve (cubic)
    """
    x = np.array(x_data)
    y = np.array(y_data)

    # --- Determine the fixed point ---
    current_fixed_point = None
    if fix_first_point and len(x) > 0:
        current_fixed_point = (x[0], y[0])

    # --- Binning and Averaging ---
    binned_x_means = []
    binned_y_means = []
    for i in range(0, len(x), bin_size):
        x_bin, y_bin = x[i:i + bin_size], y[i:i + bin_size]
        if len(x_bin) > 0:
            binned_x_means.append(np.mean(x_bin))
            binned_y_means.append(np.mean(y_bin))

    binned_x = np.array(binned_x_means)
    binned_y = np.array(binned_y_means)

    # --- Polynomial Fitting ---
    trend_func = None
    if len(binned_x) > poly_degree:
        if current_fixed_point is None:
            # Standard polynomial fit
            coeffs = np.polyfit(binned_x, binned_y, poly_degree)
            trend_func = np.poly1d(coeffs)
        else:
            # Polynomial fit constrained to pass through the first point
            x0, y0 = current_fixed_point
            # We are fitting P(x) = y0 + c_1*(x-x0) + c_2*(x^2-x0^2) + ...
            # This can be solved with linear least squares.
            A = np.zeros((len(binned_x), poly_degree))
            for i in range(1, poly_degree + 1):
                A[:, i - 1] = binned_x ** i - x0 ** i
            
            b = binned_y - y0
            coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            def constrained_poly(x_val):
                res = y0
                for i in range(poly_degree):
                    res += coeffs[i] * (x_val ** (i + 1) - x0 ** (i + 1))
                return res
            trend_func = constrained_poly

    if trend_func is None:
        print("Warning: Not enough binned points to fit a polynomial.")
        x_smooth, y_smooth = binned_x, binned_y
    else:
        x_smooth = np.linspace(min(x), max(x), 300)
        y_smooth = trend_func(x_smooth)

    # --- Plotting ---
    ax.scatter(x, y, marker='s', alpha=1.0, color=marker_color, label=f"{label} (Raw Data)")
    ax.plot(x_smooth, y_smooth, color=line_color, linewidth=2.5, label=f"{label} (Trend)")


# --- Main Plotting Logic ---
def create_plots():
    """Load data from .pkl files and generate two independent plots (Pauli and HEA)."""
    pauli_file = f'pauli_variances_N{NUM_QUBITS}.pkl'
    hea_file = f'hea_variances_N{NUM_QUBITS}.pkl'

    # --- Load Pauli data ---
    if os.path.exists(pauli_file):
        with open(pauli_file, 'rb') as f:
            pauli_data = pickle.load(f)
            pauli_layers = pauli_data.get('layers', [])
            pauli_variances = pauli_data.get('variances', [])
        print(f"Loaded {len(pauli_layers)} data points for Pauli Ansatz.")
    else:
        print(f"Error: Data file not found: {pauli_file}")
        pauli_layers, pauli_variances = [], []

    # --- Load HEA data ---
    if os.path.exists(hea_file):
        with open(hea_file, 'rb') as f:
            hea_data = pickle.load(f)
            hea_layers = hea_data.get('layers', [])
            hea_variances = hea_data.get('variances', [])
        print(f"Loaded {len(hea_layers)} data points for HEA.")
    else:
        print(f"Error: Data file not found: {hea_file}")
        hea_layers, hea_variances = [], []

    if not pauli_layers and not hea_layers:
        print("No data available to plot.")
        return

    # --- Pauli plot ---
    if pauli_layers:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_binned_average(
            ax, pauli_layers[5:], pauli_variances[5:],
            # ----> TRY CHANGING THIS NUMBER (e.g., 2, 3, or 4) <----
            poly_degree=3,
            label='Pauli Ansatz', marker_color='orange', line_color='red'
        )
        ax.set_xlabel("Number of Layers")
        ax.set_ylabel("Gradient Variance")
        ax.set_yscale("log")
        ax.set_title(f"Gradient Variance vs. Circuit Depth ({NUM_QUBITS} Qubits) - Pauli Ansatz")
        ax.legend()
        ax.grid(True, which="both", ls="--")
        plot_filename = f'variance_comparison_N{NUM_QUBITS}_pauli'
        plt.savefig(f'{plot_filename}.eps', format='eps', bbox_inches='tight')
        plt.savefig(f'{plot_filename}.png', format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}.eps and {plot_filename}.png")
        plt.show()

    # --- HEA plot ---
    if hea_layers:
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_binned_average(
            ax, hea_layers[1:], hea_variances[1:],
            # ----> TRY CHANGING THIS NUMBER (e.g., 2, 3, or 4) <----
            poly_degree=3,
            label='HEA', marker_color='orange', line_color='red'
        )
        ax.set_xlabel("Number of Layers")
        ax.set_ylabel("Gradient Variance")
        ax.set_yscale("log")
        ax.set_title(f"Gradient Variance vs. Circuit Depth ({NUM_QUBITS} Qubits) - HEA")
        ax.legend()
        ax.grid(True, which="both", ls="--")
        plot_filename = f'variance_comparison_N{NUM_QUBITS}_hea'
        plt.savefig(f'{plot_filename}.eps', format='eps', bbox_inches='tight')
        plt.savefig(f'{plot_filename}.png', format='png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as {plot_filename}.eps and {plot_filename}.png")
        plt.show()


if __name__ == "__main__":
    create_plots()