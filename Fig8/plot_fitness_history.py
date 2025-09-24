import pandas as pd
import matplotlib.pyplot as plt
import os

def load_fitness_data(filename: str):
    """
    Loads the fitness history from a CSV file into a pandas DataFrame.

    Args:
        filename (str): The path to the input CSV file.

    Returns:
        pandas.DataFrame: A DataFrame containing the fitness data, or None if the file is not found.
    """
    if not os.path.isfile(filename):
        print(f"--- Error ---")
        print(f"File not found at: '{filename}'")
        print(f"Please make sure the CSV file exists and the filename is correct.")
        print(f"---")
        return Nonez

    print(f"Loading data from '{filename}'...")
    # Use pandas to read the CSV. We set the first column ('Num_Qubits')
    # as the index of the DataFrame, which makes plotting easier.
    data = pd.read_csv(filename, index_col='Num_Qubits')
    return data

def create_fitness_plot(fitness_df: pd.DataFrame, save_filename: str):
    """
    Generates and saves a plot from the fitness history DataFrame.

    Args:
        fitness_df (pd.DataFrame): The DataFrame loaded from the CSV.
        save_filename (str): The path to save the output plot image.
    """
    if fitness_df is None or fitness_df.empty:
        print("No data available to plot.")
        return

    print(f"Generating fitness history plot...")
    plt.style.use('seaborn-v0_8-whitegrid') # Using a nice style for the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Iterate over each row in the DataFrame.
    # The 'index' will be the number of qubits (e.g., 2, 3, 4...).
    # The 'history_series' will be the list of fitness values for that row.
    for num_qubits, history_series in fitness_df.iterrows():
        # .dropna() handles cases where some runs might have had fewer generations than others.
        history = history_series.dropna().values
        generations = range(1, len(history) + 1)
        ax.plot(generations, history, marker='o', linestyle='-', markersize=5, label=f'{num_qubits} Qubits')

    ax.set_title('GA Fitness vs. Generation', fontsize=16, weight='bold')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel(r'$\frac{1}{d} \sum_{i=1}^{d} \left| \frac{\partial C}{\partial \theta_i} \right|$', fontsize=14)
    #ax.set_yscale('log') # Use a logarithmic scale to better see improvements
    ax.legend(title="Number of Qubits")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.savefig(save_filename[:-4]+'.eps', format= 'eps')
    print(f"Fitness history plot successfully saved to '{save_filename}'")

    # Also display the plot on screen
    plt.show()

if __name__ == "__main__":
    # --- Configuration ---
    # This filename MUST match the output file from your main experiment script.
    INPUT_CSV_FILE = "ga_fitness.csv"
    OUTPUT_PLOT_FILE = "fitness_history.png"

    # 1. Load the data from the CSV file
    fitness_data = load_fitness_data(INPUT_CSV_FILE)

    # 2. If data was loaded successfully, create the plot
    if fitness_data is not None:
        create_fitness_plot(fitness_data, OUTPUT_PLOT_FILE)