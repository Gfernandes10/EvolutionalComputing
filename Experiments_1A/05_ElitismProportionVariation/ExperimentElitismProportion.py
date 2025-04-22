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
    # Define the elitism proportions to test
    elitism_proportions = [0.1, 0.2, 0.3, 0.4]  # Proportions of elites/population
    fitness_function = 'Levi'  # Example fitness function
    population_size = 200  # Fixed population size
    num_executions = 100  # Number of executions for each elitism proportion
    optimal_solution = [1, 1]  # Known optimal solution for Drop-Wave

    # Define the results directory
    results_dir = os.path.join(os.path.dirname(__file__), fitness_function)
    identifier_prefix = fitness_function + 'Exp'

    # Run the experiments
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists
    for proportion in elitism_proportions:
        num_elites = int(proportion * population_size)
        print(f"Running experiment with ELITISM_PROPORTION={proportion} (Number of Elites: {num_elites})")
        OptimizationObject = MainOptimizationScript(
            FITNESS_FUNCTION_SELECTION=fitness_function,
            IDENTIFIER=f"{identifier_prefix}_EP{int(proportion * 100)}"
        )
        OptimizationObject.POPULATION_SIZE = population_size
        OptimizationObject.OPTIMIZATION_METHOD_NUMBER_ELITES = num_elites
        OptimizationObject.RESULTS_BASE_DIR = results_dir
        OptimizationObject.GENERATION_COUNT = 100
        OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)

    # Load results and generate plots
    data = load_performance_metrics(results_dir)
    if (data.empty):
        print("No performance metrics found.")
        return

    # Generate plots
    plot_fitness_vs_elitism_proportion(data, results_dir)
    plot_execution_time_vs_elitism_proportion(data, results_dir)
    plot_convergence_curves(results_dir)
    plot_diversity_curves(results_dir)
    plot_diversity_vs_elitism_proportion(data, results_dir)  # New function to plot diversity vs. elitism proportion
    plot_success_rate_vs_elitism_proportion(data, results_dir, optimal_solution=optimal_solution)
    
    # Define the elitism proportions to test
    elitism_proportions = [0.1, 0.2, 0.3, 0.4]  # Proportions of elites/population
    fitness_function = 'Drop-Wave'  # Example fitness function
    num_executions = 100  # Number of executions for each elitism proportion
    population_size = 200  # Fixed population size
    optimal_solution = [0, 0]  # Known optimal solution for Drop-Wave

    # Define the results directory
    results_dir = os.path.join(os.path.dirname(__file__), fitness_function)
    identifier_prefix = fitness_function + 'Exp'

    # Run the experiments
    os.makedirs(results_dir, exist_ok=True)  # Ensure the results directory exists
    for proportion in elitism_proportions:
        num_elites = int(proportion * population_size)
        print(f"Running experiment with ELITISM_PROPORTION={proportion} (Number of Elites: {num_elites})")
        OptimizationObject = MainOptimizationScript(
            FITNESS_FUNCTION_SELECTION=fitness_function,
            IDENTIFIER=f"{identifier_prefix}_EP{int(proportion * 100)}"
        )
        OptimizationObject.POPULATION_SIZE = population_size
        OptimizationObject.OPTIMIZATION_METHOD_NUMBER_ELITES = num_elites
        OptimizationObject.RESULTS_BASE_DIR = results_dir
        OptimizationObject.GENERATION_COUNT = 200
        OptimizationObject.multiple_optimization(num_executions=num_executions, optimal_solution=optimal_solution)

    # Load results and generate plots
    data = load_performance_metrics(results_dir)
    if (data.empty):
        print("No performance metrics found.")
        return

    # Generate plots
    plot_fitness_vs_elitism_proportion(data, results_dir)
    plot_execution_time_vs_elitism_proportion(data, results_dir)
    plot_convergence_curves(results_dir)
    plot_diversity_curves(results_dir)
    plot_diversity_vs_elitism_proportion(data, results_dir)  # New function to plot diversity vs. elitism proportion
    plot_success_rate_vs_elitism_proportion(data, results_dir, optimal_solution=optimal_solution)

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

def plot_fitness_vs_elitism_proportion(data, results_dir):
    """
    Plot the best fitness vs. elitism proportion.
    :param data: DataFrame containing performance metrics.
    """
    # Extract population size and number of elites
    population_data = data[data["Metric"] == "POPULATION_SIZE"][["Experiment", "Value"]].copy()
    population_data.rename(columns={"Value": "PopulationSize"}, inplace=True)
    population_data["PopulationSize"] = population_data["PopulationSize"].astype(int)

    elites_data = data[data["Metric"] == "OPTIMIZATION_METHOD_NUMBER_ELITES"][["Experiment", "Value"]].copy()
    elites_data.rename(columns={"Value": "NumElites"}, inplace=True)
    elites_data["NumElites"] = elites_data["NumElites"].astype(int)

    # Merge population size and number of elites to calculate ElitismProportion
    elites_data = elites_data.merge(population_data, on="Experiment", how="left")
    elites_data["ElitismProportion"] = elites_data["NumElites"] / elites_data["PopulationSize"]

    fitness_data = data[data["Metric"] == "Best Solution Found"].copy()
    fitness_data["Value"] = fitness_data["Value"].astype(float)  # Convert Value to float

    # Merge the two DataFrames on the "Experiment" column
    fitness_data = fitness_data.merge(elites_data, on="Experiment", how="left")

    # Group by ElitismProportion and calculate the average fitness
    avg_fitness = fitness_data.groupby("ElitismProportion")["Value"].mean().sort_index()  # Sort by ElitismProportion

    # Plot the results
    plt.figure()
    avg_fitness.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Average Best Fitness vs. Elitism Proportion")
    plt.xlabel("Elitism Proportion")
    plt.ylabel("Average Best Fitness")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fitness_vs_elitism_proportion.png"))

def plot_execution_time_vs_elitism_proportion(data, results_dir):
    """
    Plot the execution time vs. elitism proportion.
    :param data: DataFrame containing performance metrics.
    """
    # Extract population size and number of elites
    population_data = data[data["Metric"] == "POPULATION_SIZE"][["Experiment", "Value"]].copy()
    population_data.rename(columns={"Value": "PopulationSize"}, inplace=True)
    population_data["PopulationSize"] = population_data["PopulationSize"].astype(int)

    elites_data = data[data["Metric"] == "OPTIMIZATION_METHOD_NUMBER_ELITES"][["Experiment", "Value"]].copy()
    elites_data.rename(columns={"Value": "NumElites"}, inplace=True)
    elites_data["NumElites"] = elites_data["NumElites"].astype(int)

    # Merge population size and number of elites to calculate ElitismProportion
    elites_data = elites_data.merge(population_data, on="Experiment", how="left")
    elites_data["ElitismProportion"] = elites_data["NumElites"] / elites_data["PopulationSize"]

    execution_time_data = data[data["Metric"] == "Total Execution Time (s)"].copy()
    execution_time_data["Value"] = execution_time_data["Value"].astype(float)  # Convert Value to float

    # Merge the two DataFrames on the "Experiment" column
    execution_time_data = execution_time_data.merge(elites_data, on="Experiment", how="left")

    # Group by ElitismProportion and calculate the average execution time
    avg_time = execution_time_data.groupby("ElitismProportion")["Value"].mean().sort_index()  # Sort by ElitismProportion

    # Plot the results
    plt.figure()
    avg_time.plot(kind="bar", color="lightgreen", edgecolor="black")
    plt.title("Execution Time vs. Elitism Proportion")
    plt.xlabel("Elitism Proportion")
    plt.ylabel("Execution Time (s)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "execution_time_vs_elitism_proportion.png"))

def plot_diversity_vs_elitism_proportion(data, results_dir):
    """
    Plot the average diversity (standard deviation and Euclidean distance) vs. elitism proportion.
    :param data: DataFrame containing performance metrics.
    :param results_dir: Path to the directory containing experiment results.
    """
    # Extract population size and number of elites
    population_data = data[data["Metric"] == "POPULATION_SIZE"][["Experiment", "Value"]].copy()
    population_data.rename(columns={"Value": "PopulationSize"}, inplace=True)
    population_data["PopulationSize"] = population_data["PopulationSize"].astype(int)

    elites_data = data[data["Metric"] == "OPTIMIZATION_METHOD_NUMBER_ELITES"][["Experiment", "Value"]].copy()
    elites_data.rename(columns={"Value": "NumElites"}, inplace=True)
    elites_data["NumElites"] = elites_data["NumElites"].astype(int)

    # Merge population size and number of elites to calculate ElitismProportion
    elites_data = elites_data.merge(population_data, on="Experiment", how="left")
    elites_data["ElitismProportion"] = elites_data["NumElites"] / elites_data["PopulationSize"]

    diversity_data = []

    for folder in sorted(os.listdir(results_dir)):  # Sort folders for consistent order
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            diversity_path = os.path.join(folder_path, "diversity_metrics.csv")
            if os.path.exists(diversity_path):
                df = pd.read_csv(diversity_path)
                avg_std_dev = df["StdDevDiversity"].mean()  # Average standard deviation diversity
                avg_euclidean = df["EuclideanDiversity"].mean()  # Average Euclidean diversity

                # Get the elitism proportion for the current experiment
                elitism_proportion = elites_data.loc[elites_data["Experiment"] == folder, "ElitismProportion"].iloc[0]
                diversity_data.append({"ElitismProportion": elitism_proportion, "AvgStdDev": avg_std_dev, "AvgEuclidean": avg_euclidean})

    # Convert to DataFrame for plotting
    diversity_df = pd.DataFrame(diversity_data).sort_values(by="ElitismProportion")

    # Plot standard deviation diversity
    plt.figure()
    plt.plot(diversity_df["ElitismProportion"], diversity_df["AvgStdDev"], marker="o", label="Std Dev Diversity", color="blue")
    plt.plot(diversity_df["ElitismProportion"], diversity_df["AvgEuclidean"], marker="o", label="Euclidean Diversity", color="green")
    plt.title("Average Diversity vs. Elitism Proportion")
    plt.xlabel("Elitism Proportion")
    plt.ylabel("Diversity")
    plt.legend()
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_vs_elitism_proportion.png"))

def plot_diversity_curves(results_dir):
    """
    Plot diversity curves separately: standard deviation and Euclidean distance for different elitism proportions.
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
    plt.legend(title="Elitism Proportion", loc="upper right", fontsize="small")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_curves_std_dev.png"))


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
    plt.legend(title="Elitism Proportion", loc="upper right", fontsize="small")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "diversity_curves_euclidean.png"))


def plot_success_rate_vs_elitism_proportion(data, results_dir, optimal_solution=None, tolerance=1e-2):
    """
    Plot the success rate vs. elitism proportion.
    :param data: DataFrame containing performance metrics.
    :param results_dir: Directory to save the plot.
    :param optimal_solution: (Unused) Known optimal solution.
    :param tolerance: (Unused) Tolerance for determining success.
    """
    # Extract population size and number of elites
    population_data = data[data["Metric"] == "POPULATION_SIZE"][["Experiment", "Value"]].copy()
    population_data.rename(columns={"Value": "PopulationSize"}, inplace=True)
    population_data["PopulationSize"] = population_data["PopulationSize"].astype(int)

    elites_data = data[data["Metric"] == "OPTIMIZATION_METHOD_NUMBER_ELITES"][["Experiment", "Value"]].copy()
    elites_data.rename(columns={"Value": "NumElites"}, inplace=True)
    elites_data["NumElites"] = elites_data["NumElites"].astype(int)

    # Merge population size and number of elites to calculate ElitismProportion
    elites_data = elites_data.merge(population_data, on="Experiment", how="left")
    elites_data["ElitismProportion"] = elites_data["NumElites"] / elites_data["PopulationSize"]

    success_data = data[data["Metric"] == "Success Rate (%)"].copy()
    success_data["Value"] = success_data["Value"].astype(float)  # Ensure Value is numeric

    # Merge the two DataFrames on the "Experiment" column
    success_data = success_data.merge(elites_data, on="Experiment", how="left")

    # Group by ElitismProportion and directly use the success rate
    success_rate = success_data.set_index("ElitismProportion")["Value"].sort_index()  # Sort by ElitismProportion

    # Plot the results
    plt.figure()
    success_rate.plot(kind="bar", color="gold", edgecolor="black")
    plt.title("Success Rate vs. Elitism Proportion")
    plt.xlabel("Elitism Proportion")
    plt.ylabel("Success Rate (%)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "success_rate_vs_elitism_proportion.png"))

def plot_convergence_curves(results_dir):
    """
    Plot convergence curves for each elitism proportion, including the average and standard deviation.
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

    plt.title("Convergence Curves for Different Elitism Proportions")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend(title="Elitism Proportion", loc="upper right", fontsize="small")
    plt.grid(linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "convergence_curves_all_elitism_proportions.png"))

if __name__ == "__main__":
    main()
