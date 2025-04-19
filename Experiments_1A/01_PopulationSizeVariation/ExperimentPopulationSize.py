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
    # Define the population sizes to test
    population_sizes = [50, 100, 200, 400]
    fitness_function =  'Drop-Wave' #'Levi' 
    num_executions = 20  # Number of executions for each population size
    optimal_solution=[0,0] #[1, 1]

    # Define the results directory
    results_dir = os.path.join(os.path.dirname(__file__),fitness_function)
    identifier_prefix=fitness_function + 'Experiment'

    # Run the experiments
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists
    for size in population_sizes:
        print(f"Running experiment with POPULATION_SIZE={size}")
        OptimizationObject = MainOptimizationScript(
            FITNESS_FUNCTION_SELECTION=fitness_function,
            IDENTIFIER=f"{identifier_prefix}_POP{size}"
        )
        OptimizationObject.POPULATION_SIZE = size
        OptimizationObject.RESULTS_BASE_DIR = results_dir
        OptimizationObject.GENERATION_COUNT = 100
        OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution = optimal_solution)

    # Load results and generate plots
    data = load_performance_metrics(results_dir)
    if data.empty:
        print("No performance metrics found.")
        return

    # Generate plots
    plot_fitness_vs_population(data, results_dir)
    plot_execution_time_vs_population(data, results_dir)
    plot_convergence_curves(results_dir)
    plot_diversity_curves(results_dir)
    plot_diversity_vs_population(data, results_dir)  
    plot_success_rate_vs_population(data, results_dir, optimal_solution=optimal_solution)


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

def plot_fitness_vs_population(data, results_dir):
    """
    Plot the best fitness vs. population size.
    :param data: DataFrame containing performance metrics.
    """
    # Filter rows for "POPULATION_SIZE" and "Best Solution Found"
    population_sizes = data[data["Metric"] == "POPULATION_SIZE"][["Experiment", "Value"]].copy()
    population_sizes.rename(columns={"Value": "PopulationSize"}, inplace=True)
    population_sizes["PopulationSize"] = population_sizes["PopulationSize"].astype(int)

    fitness_data = data[data["Metric"] == "Best Solution Found"].copy()
    fitness_data["Value"] = fitness_data["Value"].astype(float)  # Convert Value to float

    # Merge the two DataFrames on the "Experiment" column
    fitness_data = fitness_data.merge(population_sizes, on="Experiment", how="left")

    # Group by PopulationSize and calculate the average fitness
    avg_fitness = fitness_data.groupby("PopulationSize")["Value"].mean().sort_index()  # Sort by PopulationSize

    # Plot the results
    plt.figure()
    avg_fitness.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Average Best Fitness vs. Population Size")
    plt.xlabel("Population Size")
    plt.ylabel("Average Best Fitness")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fitness_vs_population.png"))
    plt.show(block=False)

def plot_execution_time_vs_population(data, results_dir):
    """
    Plot the execution time vs. population size.
    :param data: DataFrame containing performance metrics.
    """
    # Filter rows for "POPULATION_SIZE" and "Total Execution Time (s)"
    population_sizes = data[data["Metric"] == "POPULATION_SIZE"][["Experiment", "Value"]].copy()
    population_sizes.rename(columns={"Value": "PopulationSize"}, inplace=True)
    population_sizes["PopulationSize"] = population_sizes["PopulationSize"].astype(int)

    execution_time_data = data[data["Metric"] == "Total Execution Time (s)"].copy()
    execution_time_data["Value"] = execution_time_data["Value"].astype(float)  # Convert Value to float

    # Merge the two DataFrames on the "Experiment" column
    execution_time_data = execution_time_data.merge(population_sizes, on="Experiment", how="left")

    # Group by PopulationSize and calculate the average execution time
    avg_time = execution_time_data.groupby("PopulationSize")["Value"].mean().sort_index()  # Sort by PopulationSize

    # Plot the results
    plt.figure()
    avg_time.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Execution Time vs. Population Size")
    plt.xlabel("Population Size")
    plt.ylabel("Execution Time (s)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "execution_time_vs_population.png"))
    plt.show(block=False)

def plot_convergence_curves(results_dir):
    """
    Plot convergence curves for each population size, including the average and standard deviation.
    :param results_dir: Path to the directory containing experiment results.
    """
    plt.figure()
    for folder in sorted(os.listdir(results_dir)):  # Sort folders for consistent order
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            convergence_path = os.path.join(folder_path, "convergence_curve.csv")
            if os.path.exists(convergence_path):
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

    plt.title("Convergence Curves for Different Population Sizes")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(title="Population Size", loc="upper right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "convergence_curves_all_populations.png"))
    plt.show(block=False)


def plot_diversity_curves(results_dir):
    """
    Plot diversity curves (standard deviation and Euclidean distance) for each population size.
    :param results_dir: Path to the directory containing experiment results.
    """
    plt.figure()
    for folder in sorted(os.listdir(results_dir)):  # Sort folders for consistent order
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_metrics.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                generations = df["Generation"]
                std_dev_diversity = df["StdDevDiversity"]
                euclidean_diversity = df["EuclideanDiversity"]

                # Plot standard deviation diversity
                plt.plot(generations, std_dev_diversity, label=f"{folder} (Std Dev)", linestyle="--", alpha=0.8)

                # Plot Euclidean diversity
                plt.plot(generations, euclidean_diversity, label=f"{folder} (Euclidean)", alpha=0.8)

    plt.title("Diversity Curves (Std Dev and Euclidean) for Different Population Sizes")
    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.legend(title="Population Size", loc="upper right", fontsize="small")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_curves.png"))
    plt.show(block=False)

def plot_success_rate_vs_population(data, results_dir, optimal_solution=None, tolerance=1e-2):
    """
    Plot the success rate vs. population size.
    :param data: DataFrame containing performance metrics.
    :param results_dir: Directory to save the plot.
    :param optimal_solution: (Unused) Known optimal solution.
    :param tolerance: (Unused) Tolerance for determining success.
    """
    # Filter rows for "POPULATION_SIZE" and "Success Rate (%)"
    population_sizes = data[data["Metric"] == "POPULATION_SIZE"][["Experiment", "Value"]].copy()
    population_sizes.rename(columns={"Value": "PopulationSize"}, inplace=True)
    population_sizes["PopulationSize"] = population_sizes["PopulationSize"].astype(int)

    success_data = data[data["Metric"] == "Success Rate (%)"].copy()
    success_data["Value"] = success_data["Value"].astype(float)  # Ensure Value is numeric

    # Merge the two DataFrames on the "Experiment" column
    success_data = success_data.merge(population_sizes, on="Experiment", how="left")

    # Group by PopulationSize and directly use the success rate
    success_rate = success_data.set_index("PopulationSize")["Value"].sort_index()  # Sort by PopulationSize

    # Plot the results
    plt.figure()
    success_rate.plot(kind="bar", color="gold", edgecolor="black")
    plt.title("Success Rate vs. Population Size")
    plt.xlabel("Population Size")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "success_rate_vs_population.png"))
    plt.show(block=False)

def plot_diversity_vs_population(data, results_dir):
    """
    Plot the average diversity (standard deviation and Euclidean distance) vs. population size.
    :param data: DataFrame containing performance metrics.
    :param results_dir: Path to the directory containing experiment results.
    """
    diversity_data = []

    # Extract population sizes and diversity metrics from the data
    population_sizes = data[data["Metric"] == "POPULATION_SIZE"][["Experiment", "Value"]].copy()
    population_sizes.rename(columns={"Value": "PopulationSize"}, inplace=True)
    population_sizes["PopulationSize"] = population_sizes["PopulationSize"].astype(int)

    for folder in sorted(os.listdir(results_dir)):  # Sort folders for consistent order
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_metrics.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                avg_std_dev = df["StdDevDiversity"].mean()  # Average standard deviation diversity
                avg_euclidean = df["EuclideanDiversity"].mean()  # Average Euclidean diversity

                # Get the population size for the current experiment
                population_size = population_sizes.loc[population_sizes["Experiment"] == folder, "PopulationSize"].iloc[0]
                diversity_data.append({"PopulationSize": population_size, "AvgStdDev": avg_std_dev, "AvgEuclidean": avg_euclidean})

    # Convert to DataFrame for plotting
    diversity_df = pd.DataFrame(diversity_data).sort_values(by="PopulationSize")

    # Plot standard deviation diversity
    plt.figure()
    plt.plot(diversity_df["PopulationSize"], diversity_df["AvgStdDev"], marker="o", label="Std Dev Diversity", color="blue")
    plt.plot(diversity_df["PopulationSize"], diversity_df["AvgEuclidean"], marker="o", label="Euclidean Diversity", color="green")
    plt.title("Average Diversity vs. Population Size")
    plt.xlabel("Population Size")
    plt.ylabel("Diversity")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_vs_population.png"))


if __name__ == "__main__":
    main()