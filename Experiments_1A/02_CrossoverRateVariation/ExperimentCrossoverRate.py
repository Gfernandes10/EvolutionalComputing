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
    # Define the crossover rates to test
    crossover_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
    fitness_function = 'Drop-Wave'  # Example fitness function
    num_executions = 20  # Number of executions for each crossover rate
    optimal_solution = [0, 0]  # Known optimal solution for Drop-Wave

    # Define the results directory
    results_dir = os.path.join(os.path.dirname(__file__), fitness_function + "_CrossoverRate")
    identifier_prefix = fitness_function + 'Exp'

    # Run the experiments
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists
    for rate in crossover_rates:
        print(f"Running experiment with CROSSOVER_RATE={rate}")
        OptimizationObject = MainOptimizationScript(
            FITNESS_FUNCTION_SELECTION=fitness_function,
            IDENTIFIER=f"{identifier_prefix}_CR{int(rate * 100)}"
        )
        OptimizationObject.CROSSOVER_RATE = rate
        OptimizationObject.RESULTS_BASE_DIR = results_dir
        OptimizationObject.GENERATION_COUNT = 100
        OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)

    # Load results and generate plots
    data = load_performance_metrics(results_dir)
    if data.empty:
        print("No performance metrics found.")
        return

    # Generate plots
    plot_fitness_vs_crossover_rate(data, results_dir)
    plot_execution_time_vs_crossover_rate(data, results_dir)
    plot_convergence_curves(results_dir)
    plot_diversity_curves(results_dir)
    plot_success_rate_vs_crossover_rate(data, results_dir, optimal_solution=optimal_solution)


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

def plot_fitness_vs_crossover_rate(data, results_dir):
    """
    Plot the best fitness vs. crossover rate.
    :param data: DataFrame containing performance metrics.
    """
    # Filter rows for "CROSSOVER_RATE" and "Best Solution Found"
    crossover_rates = data[data["Metric"] == "CROSSOVER_RATE"][["Experiment", "Value"]].copy()
    crossover_rates.rename(columns={"Value": "CrossoverRate"}, inplace=True)
    crossover_rates["CrossoverRate"] = crossover_rates["CrossoverRate"].astype(float)

    fitness_data = data[data["Metric"] == "Best Solution Found"].copy()
    fitness_data["Value"] = fitness_data["Value"].astype(float)  # Convert Value to float

    # Merge the two DataFrames on the "Experiment" column
    fitness_data = fitness_data.merge(crossover_rates, on="Experiment", how="left")

    # Group by CrossoverRate and calculate the average fitness
    avg_fitness = fitness_data.groupby("CrossoverRate")["Value"].mean().sort_index()  # Sort by CrossoverRate

    # Plot the results
    plt.figure()
    avg_fitness.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Average Best Fitness vs. Crossover Rate")
    plt.xlabel("Crossover Rate")
    plt.ylabel("Average Best Fitness")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fitness_vs_crossover_rate.png"))
    plt.show(block=False)

def plot_execution_time_vs_crossover_rate(data, results_dir):
    """
    Plot the execution time vs. crossover rate.
    :param data: DataFrame containing performance metrics.
    """
    # Filter rows for "CROSSOVER_RATE" and "Total Execution Time (s)"
    crossover_rates = data[data["Metric"] == "CROSSOVER_RATE"][["Experiment", "Value"]].copy()
    crossover_rates.rename(columns={"Value": "CrossoverRate"}, inplace=True)
    crossover_rates["CrossoverRate"] = crossover_rates["CrossoverRate"].astype(float)

    execution_time_data = data[data["Metric"] == "Total Execution Time (s)"].copy()
    execution_time_data["Value"] = execution_time_data["Value"].astype(float)  # Convert Value to float

    # Merge the two DataFrames on the "Experiment" column
    execution_time_data = execution_time_data.merge(crossover_rates, on="Experiment", how="left")

    # Group by CrossoverRate and calculate the average execution time
    avg_time = execution_time_data.groupby("CrossoverRate")["Value"].mean().sort_index()  # Sort by CrossoverRate

    # Plot the results
    plt.figure()
    avg_time.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Execution Time vs. Crossover Rate")
    plt.xlabel("Crossover Rate")
    plt.ylabel("Execution Time (s)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "execution_time_vs_crossover_rate.png"))
    plt.show(block=False)

def plot_convergence_curves(results_dir):
    """
    Plot convergence curves for each crossover rate, including the average and standard deviation.
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

    plt.title("Convergence Curves for Different Crossover Rates")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(title="Crossover Rate", loc="upper right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "convergence_curves_all_crossover_rates.png"))
    plt.show(block=False)

def plot_diversity_curves(results_dir):
    """
    Plot diversity curves for each crossover rate.
    :param results_dir: Path to the directory containing experiment results.
    """
    plt.figure()
    for folder in sorted(os.listdir(results_dir)):  # Sort folders for consistent order
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_curve.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                generations = df["Generation"]
                diversity = df["Value"]
                plt.plot(generations, diversity, label=folder)
    plt.title("Diversity Curves")
    plt.xlabel("Generation")
    plt.ylabel("Diversity")
    plt.legend(title="Crossover Rate", loc="upper right")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_curves.png"))
    plt.show(block=False)

def plot_success_rate_vs_crossover_rate(data, results_dir, optimal_solution=None, tolerance=1e-2):
    """
    Plot the success rate vs. crossover rate.
    :param data: DataFrame containing performance metrics.
    :param results_dir: Directory to save the plot.
    :param optimal_solution: (Unused) Known optimal solution.
    :param tolerance: (Unused) Tolerance for determining success.
    """
    # Filter rows for "CROSSOVER_RATE" and "Success Rate (%)"
    crossover_rates = data[data["Metric"] == "CROSSOVER_RATE"][["Experiment", "Value"]].copy()
    crossover_rates.rename(columns={"Value": "CrossoverRate"}, inplace=True)
    crossover_rates["CrossoverRate"] = crossover_rates["CrossoverRate"].astype(float)

    success_data = data[data["Metric"] == "Success Rate (%)"].copy()
    success_data["Value"] = success_data["Value"].astype(float)  # Ensure Value is numeric

    # Merge the two DataFrames on the "Experiment" column
    success_data = success_data.merge(crossover_rates, on="Experiment", how="left")

    # Group by CrossoverRate and directly use the success rate
    success_rate = success_data.set_index("CrossoverRate")["Value"].sort_index()  # Sort by CrossoverRate

    # Plot the results
    plt.figure()
    success_rate.plot(kind="bar", color="gold", edgecolor="black")
    plt.title("Success Rate vs. Crossover Rate")
    plt.xlabel("Crossover Rate")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "success_rate_vs_crossover_rate.png"))
    plt.show(block=False)


if __name__ == "__main__":
    main()
