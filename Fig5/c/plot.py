import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import stats

def load_gradients_csv(filename):
    """Loads numerical data from a CSV file into a list of lists."""
    loaded_data = []
    try:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Ensure row is not empty
                    try:
                        loaded_data.append([float(val) for val in row])
                    except ValueError:
                        print(f"Warning: Row with non-float value skipped: {row}")
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
    except IOError as e:
        print(f"Error reading from {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during load: {e}")
    return loaded_data

def plot_fit(x_data, y_data):
    """
    Calculates an exponential fit and plots the original data and the fitted curve.
    """
    if len(x_data) != len(y_data):
        print("Error: x_data and y_data must have the same number of elements.")
        return
    if len(x_data) < 2:
        print("Error: At least two data points are required for a fit.")
        return
    if not all(y > 0 for y in y_data):
        print("Error: All y_data must be positive for exponential fit.")
        return

    x = np.array(x_data, dtype=float)
    y = np.array(y_data, dtype=float)
    log_y = np.log(y)

    slope_lin, intercept_lin, _, _, _ = stats.linregress(x, log_y)
    a = np.exp(intercept_lin)
    b = slope_lin

    fit_function = lambda val: a * np.exp(b * val)
    x_fit = np.linspace(min(x), max(x), 200)
    y_fit = fit_function(x_fit)

    plt.plot(x_fit, y_fit, color='orangered', linewidth=2, label=f'Fit Slope={b:.2f}')

# --- Main Plotting Logic ---

# Define the range of qubits used
qubit_range = list(range(2, 11))

# Load all necessary data for plotting
gradient_list = load_gradients_csv('gradient_data_5c.csv')
# Load the pre-calculated variances and probabilities (flatten the list from CSV)
variances = load_gradients_csv('calculated_variances.csv')[0]
probabilities = load_gradients_csv('calculated_probabilities.csv')[0]


# Create the plot
plt.figure(figsize=(10, 6))

# Plot raw gradients
for num_qubits, gradients in zip(qubit_range, gradient_list):
    plt.scatter(
        [num_qubits] * len(gradients),
        gradients,
        alpha=0.5,
        color="blue",
        label="Gradients" if num_qubits == qubit_range[0] else ""
    )

# Plot probability
plt.plot(
    qubit_range,
    probabilities,
    marker='o',
    color="green",
    label="Probability"
)

# Plot variance
plt.scatter(
    qubit_range,
    variances,
    marker='s',
    s=80, # Make squares bigger
    color="orange",
    label="Variance"
)

# Plot exponential fit for the variance
if variances and all(v > 0 for v in variances):
    plot_fit(qubit_range, variances)

# Configure and save the plot
plt.xlabel("Number of Qubits")
plt.ylabel("Value (log scale)")
plt.yscale("log")
plt.title('HEA Gradient Analysis')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('5c_plot.eps', format='eps')
plt.savefig('5c_plot.png')
plt.show()