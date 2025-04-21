import sys
import os
# Add the root directory to the system path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)
from MainOptimizationScript import MainOptimizationScript
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Define the mutation rates to test
    mutation_rates = [0.1, 0.3, 0.5, 0.7]
    fitness_function = 'Drop-Wave'  # Example fitness function
    num_executions = 100  # Number of executions for each mutation rate
    optimal_solution = [0, 0]  # Known optimal solution for Drop-Wave

    # Define the results directory
    results_dir = os.path.join(os.path.dirname(__file__), fitness_function + "_MutationRate")
    identifier_prefix = fitness_function + 'Experiment'

    # Run the experiments
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists
    for rate in mutation_rates:
        print(f"Running experiment with MUTATION_RATE={rate}")
        OptimizationObject = MainOptimizationScript(
            FITNESS_FUNCTION_SELECTION=fitness_function,
            IDENTIFIER=f"{identifier_prefix}_MR{int(rate * 100)}"
        )
        OptimizationObject.MUTATION_RATE = rate
        OptimizationObject.RESULTS_BASE_DIR = results_dir
        OptimizationObject.GENERATION_COUNT = 100
        OptimizationObject.APPLY_DIVERSITY_MAINTENANCE = False  # Disable diversity maintenance
        OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)
        
    # Load results and generate plots
    data = load_performance_metrics(results_dir)
    if (data.empty):
        print("No performance metrics found.")
        return

    # Generate plots
    plot_fitness_vs_mutation_rate(data, results_dir)
    plot_execution_time_vs_mutation_rate(data, results_dir)
    plot_convergence_curves(results_dir)
    plot_diversity_curves(results_dir)
    plot_diversity_vs_mutation_rate(data, results_dir)  # New function to plot diversity vs. mutation rate
    plot_success_rate_vs_mutation_rate(data, results_dir, optimal_solution=optimal_solution)


def load_performance_metrics(results_dir):
    """
    Load performance metrics from all subdirectories in the results directory.
    :param results_dir: Path to the directory containing experiment results.
    :return: DataFrame with combined performance metrics.
    """
    data = []
    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            metrics_path = os.path.join(folder_path, "performance_metrics.csv")
            if os.path.exists(metrics_path):
                try:
                    # Use on_bad_lines='skip' to ignore problematic lines
                    df = pd.read_csv(metrics_path, skip_blank_lines=True, on_bad_lines='skip')
                    df["Experiment"] = folder  # Add experiment identifier
                    data.append(df)
                except pd.errors.ParserError as e:
                    print(f"Error reading {metrics_path}: {e}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

def plot_fitness_vs_mutation_rate(data, results_dir):
    """
    Plot the best fitness vs. mutation rate.
    :param data: DataFrame containing performance metrics.
    """
    # Filter rows for "MUTATION_RATE" and "Best Solution Found"
    mutation_rates = data[data["Metric"] == "MUTATION_RATE"][["Experiment", "Value"]].copy()
    mutation_rates.rename(columns={"Value": "MutationRate"}, inplace=True)
    mutation_rates["MutationRate"] = mutation_rates["MutationRate"].astype(float)

    fitness_data = data[data["Metric"] == "Best Solution Found"].copy()
    fitness_data["Value"] = fitness_data["Value"].astype(float)  # Convert Value to float

    # Merge the two DataFrames on the "Experiment" column
    fitness_data = fitness_data.merge(mutation_rates, on="Experiment", how="left")

    # Group by MutationRate and calculate the average fitness
    avg_fitness = fitness_data.groupby("MutationRate")["Value"].mean().sort_index()  # Sort by MutationRate

    # Plot the results
    plt.figure()
    avg_fitness.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Average Best Fitness vs. Mutation Rate")
    plt.xlabel("Mutation Rate")
    plt.ylabel("Average Best Fitness")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fitness_vs_mutation_rate.png"))

def plot_execution_time_vs_mutation_rate(data, results_dir):
    """
    Plot the execution time vs. mutation rate.
    :param data: DataFrame containing performance metrics.
    """
    # Filter rows for "MUTATION_RATE" and "Total Execution Time (s)"
    mutation_rates = data[data["Metric"] == "MUTATION_RATE"][["Experiment", "Value"]].copy()
    mutation_rates.rename(columns={"Value": "MutationRate"}, inplace=True)
    mutation_rates["MutationRate"] = mutation_rates["MutationRate"].astype(float)

    execution_time_data = data[data["Metric"] == "Total Execution Time (s)"].copy()
    execution_time_data["Value"] = execution_time_data["Value"].astype(float)  # Convert Value to float

    # Merge the two DataFrames on the "Experiment" column
    execution_time_data = execution_time_data.merge(mutation_rates, on="Experiment", how="left")

    # Group by MutationRate and calculate the average execution time
    avg_time = execution_time_data.groupby("MutationRate")["Value"].mean().sort_index()  # Sort by MutationRate

    # Plot the results
    plt.figure()
    avg_time.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Execution Time vs. Mutation Rate")
    plt.xlabel("Mutation Rate")
    plt.ylabel("Execution Time (s)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "execution_time_vs_mutation_rate.png"))

def plot_convergence_curves(results_dir):
    """
    Plot convergence curves for each mutation rate, including the average and standard deviation.
    :param results_dir: Path to the directory containing experiment results.
    """
    plt.figure()
    for folder in sorted(os.listdir(results_dir)):  # Sort folders for consistent order
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            convergence_path = os.path.join(folder_path, "convergence_curve.csv")
            if os.path.exists(convergence_path):
                try:
                    df = pd.read_csv(convergence_path)
                    generations = df["Generation"]
                    avg_convergence = df["Value"]
                    std_convergence = df["StdDev"] if "StdDev" in df.columns else None

                    # Plot the average convergence curve
                    plt.plot(generations, avg_convergence, label=f"{folder} (Avg)", alpha=0.8)

                    # Plot the standard deviation as a shaded area
                    if std_convergence is not None:
                        plt.fill_between(
                            generations,
                            avg_convergence - std_convergence,
                            avg_convergence + std_convergence,
                            alpha=0.2,
                            label=f"{folder} (Std Dev)"
                        )
                except Exception as e:
                    print(f"Error processing {convergence_path}: {e}")

    plt.title("Convergence Curves for Different Mutation Rates")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(title="Mutation Rate", loc="upper right", fontsize="small")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "convergence_curves_all_mutation_rates.png"))

def plot_diversity_curves(results_dir):
    """
    Plot diversity curves separately: standard deviation and Euclidean distance for different mutation rates.
    """
    # Standard Deviation Diversity Curves
    plt.figure()
    for folder in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_metrics.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                generations = df["Generation"]
                std_dev_diversity = df["StdDevDiversity"]
                plt.plot(generations, std_dev_diversity, label=f"{folder}", linestyle="--", alpha=0.8)
    plt.title("Standard Deviation Diversity Curves")
    plt.xlabel("Generation")
    plt.ylabel("Std Dev Diversity")
    plt.legend(title="Mutation Rate", loc="upper right", fontsize="small")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_curves_std_dev.png"))
    plt.show(block=False)

    # Euclidean Diversity Curves
    plt.figure()
    for folder in sorted(os.listdir(results_dir)):
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_metrics.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                generations = df["Generation"]
                euclidean_diversity = df["EuclideanDiversity"]
                plt.plot(generations, euclidean_diversity, label=f"{folder}", alpha=0.8)
    plt.title("Euclidean Diversity Curves")
    plt.xlabel("Generation")
    plt.ylabel("Euclidean Diversity")
    plt.legend(title="Mutation Rate", loc="upper right", fontsize="small")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_curves_euclidean.png"))
    plt.show(block=False)

def plot_success_rate_vs_mutation_rate(data, results_dir, optimal_solution=None, tolerance=1e-2):
    """
    Plot the success rate vs. mutation rate.
    :param data: DataFrame containing performance metrics.
    :param results_dir: Directory to save the plot.
    :param optimal_solution: (Unused) Known optimal solution.
    :param tolerance: (Unused) Tolerance for determining success.
    """
    # Filter rows for "MUTATION_RATE" and "Success Rate (%)"
    mutation_rates = data[data["Metric"] == "MUTATION_RATE"][["Experiment", "Value"]].copy()
    mutation_rates.rename(columns={"Value": "MutationRate"}, inplace=True)
    mutation_rates["MutationRate"] = mutation_rates["MutationRate"].astype(float)

    success_data = data[data["Metric"] == "Success Rate (%)"].copy()
    success_data["Value"] = success_data["Value"].astype(float)  # Ensure Value is numeric

    # Merge the two DataFrames on the "Experiment" column
    success_data = success_data.merge(mutation_rates, on="Experiment", how="left")

    # Group by MutationRate and directly use the success rate
    success_rate = success_data.set_index("MutationRate")["Value"].sort_index()  # Sort by MutationRate

    # Plot the results
    plt.figure()
    success_rate.plot(kind="bar", color="gold", edgecolor="black")
    plt.title("Success Rate vs. Mutation Rate")
    plt.xlabel("Mutation Rate")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "success_rate_vs_mutation_rate.png"))

def plot_diversity_vs_mutation_rate(data, results_dir):
    """
    Plot the average diversity (standard deviation and Euclidean distance) vs. mutation rate.
    :param data: DataFrame containing performance metrics.
    :param results_dir: Path to the directory containing experiment results.
    """
    diversity_data = []

    # Extract mutation rates and diversity metrics from the data
    mutation_rates = data[data["Metric"] == "MUTATION_RATE"][["Experiment", "Value"]].copy()
    mutation_rates.rename(columns={"Value": "MutationRate"}, inplace=True)
    mutation_rates["MutationRate"] = mutation_rates["MutationRate"].astype(float)

    for folder in sorted(os.listdir(results_dir)):  # Sort folders for consistent order
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_metrics.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                avg_std_dev = df["StdDevDiversity"].mean()  # Average standard deviation diversity
                avg_euclidean = df["EuclideanDiversity"].mean()  # Average Euclidean diversity

                # Get the mutation rate for the current experiment
                mutation_rate = mutation_rates.loc[mutation_rates["Experiment"] == folder, "MutationRate"].iloc[0]
                diversity_data.append({"MutationRate": mutation_rate, "AvgStdDev": avg_std_dev, "AvgEuclidean": avg_euclidean})

    # Convert to DataFrame for plotting
    diversity_df = pd.DataFrame(diversity_data).sort_values(by="MutationRate")

    # Plot standard deviation diversity
    plt.figure()
    plt.plot(diversity_df["MutationRate"], diversity_df["AvgStdDev"], marker="o", label="Std Dev Diversity", color="blue")
    plt.plot(diversity_df["MutationRate"], diversity_df["AvgEuclidean"], marker="o", label="Euclidean Diversity", color="green")
    plt.title("Average Diversity vs. Mutation Rate")
    plt.xlabel("Mutation Rate")
    plt.ylabel("Diversity")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_vs_mutation_rate.png"))

if __name__ == "__main__":
    main()
