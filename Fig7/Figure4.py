import os
import csv
import pickle
import warnings
from numpy.random import default_rng

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter

# --- Configuration ---
END_LAYER = 200
NUM_QUBITS = 2
NUM_SAMPLES = 100  # Number of random points to estimate variance
SEED = 42  # A seed for all random processes for reproducibility

# Suppress warnings from Qiskit
warnings.filterwarnings('ignore', category=UserWarning, module='qiskit')
# Seed the global NumPy random generator for parameter initialization
np.random.seed(SEED)


# --- Helper Functions (Loading, Saving, Math) ---

# --- Functions for handling .pkl files ---

def load_progress_pkl(filename):
    """Loads previously computed layers and variances from a .pkl file."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            return data.get('layers', []), data.get('variances', [])
    return [], []


def save_progress_pkl(filename, layers_data, variances_data):
    """Saves the complete list of layers and variances to a .pkl file."""
    data = {'layers': layers_data, 'variances': variances_data}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


# (CSV helpers kept for compatibility / inspection)

def load_progress_csv(filename):
    """Loads previously computed layers and variances from a CSV file."""
    layers, variances = [], []
    if os.path.exists(filename):
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                next(reader)  # Skip header
                for row in reader:
                    layers.append(int(row[0]))
                    variances.append(float(row[1]))
            except StopIteration:
                pass
    return layers, variances


def save_progress_csv(filename, layer, variance):
    """Appends a new data point to the CSV file."""
    file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Layer', 'Variance'])
        writer.writerow([layer, variance])


def expectation_value(qc, params):
    """Estimate expectation value of ZZ between qubit 0 and 1 for the bound circuit."""
    # Build a ZZ 
    if qc.num_qubits < 2:
        raise ValueError("Need at least 2 qubits for the ZZ observable")
    pauli_label = "ZZ" + "I" * (qc.num_qubits - 2)
    hamiltonian = SparsePauliOp(pauli_label)
    bound_qc = qc.assign_parameters(params)
    estimator = Estimator()
    job = estimator.run(circuits=[bound_qc], observables=[hamiltonian])
    return job.result().values[0]


def two_point_gradient(qc, params, cost_func, step_size=1e-3, param_index=0):
    """Computes the partial derivative using the two-point central finite difference formula.

    Note: `params` is expected to be a NumPy array-like of the same length as qc.num_parameters.
    This function computes derivative with respect to the parameter at `param_index`.
    """
    params_plus = np.copy(params)
    params_minus = np.copy(params)
    params_plus[param_index] += step_size
    params_minus[param_index] -= step_size
    cost_plus = cost_func(qc, params_plus)
    cost_minus = cost_func(qc, params_minus)
    return (cost_plus - cost_minus) / (2 * step_size)


# --- Ansatz Definitions ---

def manual_pauli_ansatz_ring(num_qubits, depth=1, seed=None):
    """
    Manually builds a Pauli-based ansatz with a non-repeating initial layer
    and a seeded structure for reproducibility.
    """
    rng = default_rng(seed)  # Create a seeded random number generator
    qc = QuantumCircuit(num_qubits)

    # 1. Initial, non-repeating RY layer
    for i in range(num_qubits):
        qc.ry(np.pi / 4, i)

    # 2. Repeating blocks
    for d in range(depth):
        layer_params = ParameterVector(f'p_{d}', length=num_qubits)
        # Seeded random choice of Pauli gates for this layer
        pauli_gates = rng.choice(['rx', 'ry', 'rz'], size=num_qubits)
        for i in range(num_qubits):
            gate_type = pauli_gates[i]
            if gate_type == 'rx':
                qc.rx(layer_params[i], i)
            elif gate_type == 'ry':
                qc.ry(layer_params[i], i)
            else:  # rz
                qc.rz(layer_params[i], i)

        # Entangling layer (linear chain using CZ)
        for i in range(num_qubits - 1):
            qc.cz(i, i + 1)

        qc.barrier()

    return qc


def hea_ansatz_ring(num_qubits, depth=1):
    """Hardware-efficient ansatz with a ring connection."""
    qc = QuantumCircuit(num_qubits)
    num_params = num_qubits * 3 * depth
    theta = [Parameter(f'θ_{i}') for i in range(num_params)]
    param_index = 0
    for d in range(depth):
        for q in range(num_qubits):
            qc.ry(theta[param_index], q)
            qc.rx(theta[param_index + 1], q)
            qc.ry(theta[param_index + 2], q)
            param_index += 3
        if num_qubits > 1:
            for q1 in range(num_qubits - 1):
                qc.cz(q1, q1 + 1)
    return qc


# --- Main Analysis Loop ---

def run_analysis(ansatz_type, end_layer, pkl_filename, title):
    """Main function to run analysis for a given ansatz type."""
    print(f"--- Starting analysis for: {title} ---")

    layers_done, variances_done = load_progress_pkl(pkl_filename)
    start_layer = max(layers_done) + 1 if layers_done else 1

    print(f"Loaded {len(layers_done)} points. Starting from layer {start_layer} to {end_layer}.")

    if start_layer <= end_layer:
        for layers in range(start_layer, end_layer + 1):
            if ansatz_type == 'manual_pauli':
                qc = manual_pauli_ansatz_ring(num_qubits=NUM_QUBITS, depth=layers, seed=SEED)
            elif ansatz_type == 'hea_ring':
                qc = hea_ansatz_ring(num_qubits=NUM_QUBITS, depth=layers)
            else:
                raise ValueError(f"Unknown ansatz_type: {ansatz_type}")

            gradients = []
            if qc.num_parameters > 0:
                for _ in range(NUM_SAMPLES):
                    params = np.random.uniform(0, 2 * np.pi, qc.num_parameters)
                    # Compute gradient of the first parameter as a sample (you can randomize index if desired)
                    grad = two_point_gradient(qc, params, cost_func=expectation_value, param_index=0)
                    gradients.append(grad)

            variance = float(np.var(gradients)) if gradients else 0.0

            # Append to lists and save all data each time
            layers_done.append(layers)
            variances_done.append(variance)
            save_progress_pkl(pkl_filename, layers_done, variances_done)

            print(f"Layer: {layers}, Variance: {variance} (saved)")
    else:
        print("All target layers already computed.")

    print(f"--- Analysis complete for {title} ---")


# --- Run Both Analyses ---
if __name__ == "__main__":
    run_analysis(
        ansatz_type='manual_pauli',
        end_layer=END_LAYER,
        pkl_filename=f'pauli_variances_N{NUM_QUBITS}.pkl',
        title=f'Pauli Ansatz ({NUM_QUBITS} Qubits)'
    )

    run_analysis(
        ansatz_type='hea_ring',
        end_layer=END_LAYER,
        pkl_filename=f'hea_variances_N{NUM_QUBITS}.pkl',
        title=f'HEA ({NUM_QUBITS} Qubits)'
    )
