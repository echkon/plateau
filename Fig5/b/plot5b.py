import pickle
import matplotlib.pyplot as plt

# Load the data from the .pkl file
with open('pauli_plot_data.pkl', 'rb') as f:
    plot_data = pickle.load(f)

# Extract the data from the loaded dictionary
theta_vals = plot_data['theta_vals']
results = plot_data['results']

# Plot each data series
for result in results:
    n = result['n']
    exp_val = result['exp_val']
    plt.plot(theta_vals, exp_val, label=f'N={n}')

# --- Add plot labels, title, and save the figure ---
plt.legend()
plt.title('PauliTwoDesign')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$C(\mathbf{\theta})$')
plt.grid(True)
plt.tight_layout()
plt.savefig('5b.eps', format='eps')
plt.savefig('5b')
plt.show()

print("Plot generated and saved as 4b.eps")