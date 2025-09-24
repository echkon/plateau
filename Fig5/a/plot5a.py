import pickle
import matplotlib.pyplot as plt

# Load the data from the .pkl file
# 'rb' means "read bytes"
with open('plot_data.pkl', 'rb') as f:
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
plt.title('HEA')
plt.xlabel(r'$\theta_1$')
plt.ylabel(r'$C(\mathbf{\theta})$')
plt.grid(True)
plt.tight_layout()
plt.savefig('5a.eps', format='eps')
plt.savefig('5a')
plt.show()

print("Plot generated and saved as 4a.eps")