import numpy as np
import pickle
import csv
from scipy import stats
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library import PauliTwoDesign

# --- Functions from Figures.ipynb ---

def hea_ansatz(num_qubits, depth=20):
    """Creates a hardware-efficient ansatz."""
    theta = [Parameter(f'θ[{i}]') for i in range(num_qubits * 3 * depth)]
    qc = QuantumCircuit(num_qubits)
    index = 0
    for d in range(depth):
        for q in range(num_qubits):
            qc.ry(theta[index], q)
            qc.rx(theta[index + 1], q)
            qc.ry(theta[index + 2], q)
            index += 3
        for q1 in range(num_qubits - 1):
            qc.cz(q1, q1 + 1)
    return qc

def get_qc(qc):
    """Helper function to rebuild a circuit with Qiskit Parameters."""
    new_qc = QuantumCircuit(qc.num_qubits)
    params = []
    # Using a different way to get qubit index to avoid deprecated .index
    qubit_map = {qubit: i for i, qubit in enumerate(qc.qubits)}
    for instr, qargs, cargs in qc.data:
        if instr.name in ['rx', 'ry', 'rz']:
            theta = Parameter(f"theta_{len(params)}")
            params.append(theta)
            qubit_index = qubit_map[qargs[0]]
            if instr.name == 'rx':
                new_qc.rx(theta, qubit_index)
            elif instr.name == 'ry':
                new_qc.ry(theta, qubit_index)
            elif instr.name == 'rz':
                new_qc.rz(theta, qubit_index)
        else:
            new_qc.data.append((instr, qargs, cargs))
    return new_qc

def pauli_ansatz(num_qubits, depth=20):
    """Creates a PauliTwoDesign-like ansatz."""
    p2d = PauliTwoDesign(num_qubits=num_qubits, reps=depth).decompose()
    qc = get_qc(p2d)
    return qc

def expectation_value(qc, params, type=None):
    """Computes the expectation value for the Hamiltonian H = Z1*Z2."""
    ham = SparsePauliOp("ZZ" + "I" * (qc.num_qubits - 2))
    qc_bound = qc.assign_parameters(params)
    
    if type == 'rpa':
        initial_layer = QuantumCircuit(qc.num_qubits)
        for i in range(qc.num_qubits):
            initial_layer.ry(np.pi / 4, i)
        qc_final = initial_layer.compose(qc_bound)
    else:
        qc_final = qc_bound
        
    sampler = Estimator()
    job = sampler.run(circuits=[qc_final], observables=[ham])
    result = job.result()
    return result.values[0]

def two_point_gradient(qc, params, cost_func, step_size, idx, type=None):
    """Two-point method to calculate the gradient."""
    params_plus = np.copy(params)
    params_minus = np.copy(params)
    
    params_plus[idx] += step_size
    params_minus[idx] -= step_size
    
    f_plus = cost_func(qc, params_plus, type=type)
    f_minus = cost_func(qc, params_minus, type=type)
    
    gradient = (f_plus - f_minus) / (2 * step_size)
    return gradient

def get_exp_fit_slope(x_data, y_data):
    """Performs a linear regression on the log of y_data to find the exponential slope."""
    log_y_data = np.log(y_data)
    slope, _, _, _, _ = stats.linregress(x_data, log_y_data)
    return slope

# --- Main Data Generation Function ---

def generate_data_for_plotting():
    """
    Generates and saves all data needed for plot(1).py into gradients_data.pkl.
    """
    print("Generating data for HEA and RPA ansätze...")
    
    qubit_range = list(range(2, 11))
    num_samples = 100
    circuit_depth = 20

    # Initialize lists to store variances for each parameter (column-wise)
    gradients_HEA = [[] for _ in range(6)]
    gradients_RPA = [[] for _ in range(6)]

    for N in qubit_range:
        print(f"  Calculating for N = {N} qubits...")
        
        # --- HEA ---
        qc_hea = hea_ansatz(N, depth=circuit_depth)
        num_params_hea = len(qc_hea.parameters)
        for k in range(6):
            if k >= num_params_hea:
                gradients_HEA[k].append(np.nan)
                continue
            grads = [two_point_gradient(qc_hea, np.random.uniform(0, 2*np.pi, num_params_hea), 
                                        expectation_value, 0.01, k, 'hea') for _ in range(num_samples)]
            gradients_HEA[k].append(np.var(grads))

        # --- RPA ---
        qc_rpa = pauli_ansatz(N, depth=circuit_depth)
        num_params_rpa = len(qc_rpa.parameters)
        for k in range(6):
            if k >= num_params_rpa:
                gradients_RPA[k].append(np.nan)
                continue
            grads = [two_point_gradient(qc_rpa, np.random.uniform(0, 2*np.pi, num_params_rpa), 
                                        expectation_value, 0.01, k, 'rpa') for _ in range(num_samples)]
            gradients_RPA[k].append(np.var(grads))

    # --- Calculate Slopes ---
    print("Calculating exponential fit slopes...")
    slopes_HEA = [get_exp_fit_slope(qubit_range, np.nan_to_num(var_data, nan=1e-10)) for var_data in gradients_HEA]
    slopes_RPA = [get_exp_fit_slope(qubit_range, np.nan_to_num(var_data, nan=1e-10)) for var_data in gradients_RPA]
    
    # --- Create dictionary and save to pickle file ---
    data_to_pickle = {
        "N": qubit_range,
        "gradients_HEA": gradients_HEA,
        "gradients_RPA": gradients_RPA,
        "slopes_HEA": slopes_HEA,
        "slopes_RPA": slopes_RPA
    }

    with open('gradients_data.pkl', 'wb') as f:
        pickle.dump(data_to_pickle, f)
        
    print("Data successfully generated and saved to gradients_data.pkl")

if __name__ == '__main__':
    generate_data_for_plotting()